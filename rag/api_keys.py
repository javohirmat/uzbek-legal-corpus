"""Per-customer API keys: bearer auth, daily request caps, usage accounting.

The Vast box is a single public port. Caddy's one shared token (AUTH=on) cannot
tell customers apart, and AUTH=off is fully open. This module gives the FastAPI
app per-key identity so a test key can be handed to a partner with a real
"free tests" limit:

    TOMARIS_API_KEYS="azizbek:t7-prod-XXXX:200,qa:t7-qa-YYYY:50"

name:key:daily_request_limit, comma-separated. Limit 0 or missing = unlimited.
With no keys configured the app stays exactly as open as today (frontend,
demo scripts, curl), so nothing breaks until keys are deliberately added.

Counters live in one JSON file (API_USAGE_FILE, default /workspace/api-usage.json
next to CHAT_LOG) so a restart does not reset the day. Every file operation is
guarded: accounting must never break a reply. /health and /v1/models stay open
so uptime probes and OpenAI-SDK reachability checks work without a key.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone


class KeyAuthError(Exception):
    """401/429 with an OpenAI-shaped error body."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": "invalid_request_error",
                          "code": self.code}}


_BEARER = re.compile(r"^\s*bearer\s+(\S+)\s*$", re.IGNORECASE)
_KEEP_DAYS = 14          # old day buckets pruned on save


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _tomorrow() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")


class KeyAuth:
    def __init__(self, keys: dict[str, dict], state_path: str):
        # secret -> {name, limit, id}; key id -> today's counters
        self._by_secret = keys
        self._state_path = state_path
        self._lock = threading.Lock()
        self._days: dict[str, dict[str, dict]] = {}
        self._load()

    # ------------------------------------------------------------ configure
    @classmethod
    def from_env(cls, env=None) -> "KeyAuth":
        env = os.environ if env is None else env
        raw = (env.get("TOMARIS_API_KEYS") or "").strip()
        # Env files and some dashboards keep the quotes literally.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            raw = raw[1:-1].strip()
        # ';' or a newline is meant as a separator but would be swallowed into
        # the secret, handing the customer a key that 401s. Never guess.
        if raw and re.search(r"[;\n\r]", raw):
            raise SystemExit(
                "[api-keys] TOMARIS_API_KEYS entries must be separated by ',' "
                "-- found ';' or a newline. Refusing to start rather than "
                "register a corrupted key.")

        keys: dict[str, dict] = {}
        seen_ids: dict[str, int] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            name, secret, limit, err = cls._parse_entry(part)
            if err:
                print(f"[api-keys] skipping malformed entry {part[:40]!r}: {err}")
                continue
            if secret in keys:
                print(f"[api-keys] duplicate secret for {name}; first definition wins")
                continue
            # Counters are stored per KEY, not per customer name: two keys for
            # one customer (a rotation, or a staging bot) must meter apart, and
            # the id must stay non-secret because it is written to disk.
            seen_ids[name] = seen_ids.get(name, 0) + 1
            key_id = name if seen_ids[name] == 1 else f"{name}#{seen_ids[name]}"
            keys[secret] = {"name": name, "limit": limit, "id": key_id}

        # A non-empty setting that produced no key used to fall through to open
        # mode: the box answers everyone, unmetered, while the operator believes
        # he just handed out a capped trial key. Empty setting = open is the
        # documented legacy behaviour and still works.
        if raw and not keys:
            raise SystemExit(
                "[api-keys] TOMARIS_API_KEYS is set but no entry parsed. "
                "Expected name:key[:daily_limit], comma-separated "
                '(e.g. TOMARIS_API_KEYS="azizbek:t7-abc123:200"). '
                "Refusing to start open and unmetered.")
        if keys:
            shown = ", ".join(f"{m['id']}({m['limit'] or 'unlimited'})"
                              for m in keys.values())
            print(f"[api-keys] {len(keys)} key(s) active: {shown}")
        else:
            print("[api-keys] OPEN MODE - no keys configured, no cap, no metering")
        path = env.get("API_USAGE_FILE") or "/workspace/api-usage.json"
        return cls(keys, path)

    @staticmethod
    def _parse_entry(part: str) -> tuple[str, str, int, str]:
        """'name:secret[:daily_limit]' -> (name, secret, limit, error).

        The secret itself may contain ':' (sk-live:abc123), so the name is
        taken from the front and the limit from the back. With three or more
        fields the last one MUST be a whole number: a typo'd limit used to read
        as 0, which means *unlimited* -- the opposite of the trial cap the
        customer was promised.
        """
        if ":" not in part:
            return "", "", 0, "no ':' separator"
        name, rest = (s.strip() for s in part.split(":", 1))
        limit = 0
        if ":" in rest:
            head, tail = (s.strip() for s in rest.rsplit(":", 1))
            if tail.isdigit():
                rest, limit = head, int(tail)
            else:
                return "", "", 0, (
                    f"daily limit {tail!r} is not a whole number "
                    "(use name:key:200, or name:key for unlimited)")
        if not name:
            return "", "", 0, "empty customer name"
        if not rest:
            return "", "", 0, "empty key"
        return name, rest, limit, ""

    @property
    def enabled(self) -> bool:
        return bool(self._by_secret)

    def limit_for(self, key_id: str) -> int:
        for meta in self._by_secret.values():
            if meta["id"] == key_id:
                return meta["limit"]
        return 0

    # ------------------------------------------------------------- persist
    def _load(self) -> None:
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("days"), dict):
                # Coerce, do not trust. Granting a customer more headroom
                # mid-day means hand-editing this file, and a slightly wrong
                # shape ({"azizbek": {"2026-08-20": 0}}) used to raise inside
                # _bucket() on EVERY request -- a total outage for the paying
                # customer while /health still reported ok.
                self._days = self._coerce(data["days"])
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[api-keys] state unreadable, starting fresh: {e}")

    _COUNTER_FIELDS = ("requests", "prompt_tokens", "completion_tokens")

    @classmethod
    def _coerce(cls, days) -> dict:
        """Keep only {key_id: {YYYY-MM-DD: {counter: int}}}; drop the rest."""
        clean: dict[str, dict[str, dict]] = {}
        for key_id, buckets in (days or {}).items():
            if not isinstance(key_id, str) or not isinstance(buckets, dict):
                print(f"[api-keys] dropping unusable usage entry {key_id!r}")
                continue
            for day, counters in buckets.items():
                if not isinstance(day, str) or not isinstance(counters, dict):
                    print(f"[api-keys] dropping unusable usage bucket {key_id!r}/{day!r}")
                    continue
                fixed = {}
                for field in cls._COUNTER_FIELDS:
                    try:
                        fixed[field] = max(0, int(counters.get(field, 0)))
                    except (TypeError, ValueError):
                        fixed[field] = 0
                clean.setdefault(key_id, {})[day] = fixed
        return clean

    def _save(self) -> None:
        self._prune()
        try:
            directory = os.path.dirname(self._state_path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".api-usage-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"days": self._days}, f, ensure_ascii=False)
            os.replace(tmp, self._state_path)
        except Exception as e:
            print(f"[api-keys] state save failed (counters stay in memory): {e}")

    def _prune(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_KEEP_DAYS)).strftime("%Y-%m-%d")
        for name in list(self._days):
            self._days[name] = {d: v for d, v in self._days[name].items() if d >= cutoff}
            if not self._days[name]:
                del self._days[name]

    # ---------------------------------------------------------------- auth
    def authorize(self, request, check_cap=True) -> str | None:
        """Validate the bearer key. Returns the key name, or None in open mode.

        Raises KeyAuthError (401 bad key, 429 over the daily limit) only when
        keys are configured -- i.e. never in the legacy open setup.
        check_cap=False identifies the caller without the limit check, so an
        over-limit key can still read /v1/usage to see when it resets.
        """
        if not self.enabled:
            return None
        header = request.headers.get("authorization") or ""
        m = _BEARER.match(header)
        if not m:
            raise KeyAuthError(401, "invalid_api_key",
                               "Missing API key. Send 'Authorization: Bearer <key>'.")
        meta = self._by_secret.get(m.group(1))
        if meta is None:
            raise KeyAuthError(401, "invalid_api_key", "Invalid API key.")
        if check_cap:
            bucket = self._bucket(meta["id"])
            limit = meta["limit"]
            if limit and bucket["requests"] >= limit:
                raise KeyAuthError(
                    429, "daily_limit_exceeded",
                    f"Daily limit of {limit} requests reached for this key. "
                    f"Resets {_tomorrow()} (UTC).")
        return meta["id"]

    def admit(self, request) -> str | None:
        """Auth + cap check + increment under one lock.

        `authorize()` then `count_request()` on two threads can both see
        remaining=1 and both proceed. Telegram bots retry. This is the
        method /v1/chat/completions and /retrieve must call.
        Open mode (no keys) returns None and does not count.
        """
        if not self.enabled:
            return None
        header = request.headers.get("authorization") or ""
        m = _BEARER.match(header)
        if not m:
            raise KeyAuthError(401, "invalid_api_key",
                               "Missing API key. Send 'Authorization: Bearer <key>'.")
        meta = self._by_secret.get(m.group(1))
        if meta is None:
            raise KeyAuthError(401, "invalid_api_key", "Invalid API key.")
        with self._lock:
            limit = meta["limit"]
            bucket = self._bucket(meta["id"])
            if limit and bucket["requests"] >= limit:
                raise KeyAuthError(
                    429, "daily_limit_exceeded",
                    f"Daily limit of {limit} requests reached for this key. "
                    f"Resets {_tomorrow()} (UTC).")
            bucket["requests"] += 1
            self._save()
        return meta["id"]

    # ------------------------------------------------------------ counters
    def _bucket(self, name: str) -> dict:
        today = _today()
        return self._days.setdefault(name, {}).setdefault(
            today, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def count_request(self, name: str | None) -> None:
        """One admitted completion request. None (open mode) is not counted."""
        if not name:
            return
        with self._lock:
            self._bucket(name)["requests"] += 1
            self._save()

    def add_usage(self, name: str | None, prompt_tokens: int,
                  completion_tokens: int) -> None:
        """Real upstream token usage, when vLLM reports it (streaming path)."""
        if not name or not (prompt_tokens or completion_tokens):
            return
        with self._lock:
            b = self._bucket(name)
            b["prompt_tokens"] += int(prompt_tokens or 0)
            b["completion_tokens"] += int(completion_tokens or 0)
            self._save()

    def usage_for(self, name: str) -> dict:
        with self._lock:
            bucket = dict(self._days.get(name, {}).get(_today(), {}))
        limit = self.limit_for(name)
        requests = bucket.get("requests", 0)
        return {
            "key": name, "date": _today(), "requests": requests,
            "daily_limit": limit or None,
            "remaining": (limit - requests) if limit else None,
            "prompt_tokens": bucket.get("prompt_tokens", 0),
            "completion_tokens": bucket.get("completion_tokens", 0),
        }
