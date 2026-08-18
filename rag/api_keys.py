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
        # secret -> {name, limit}; name -> today's counters
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
        keys: dict[str, dict] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            bits = [b.strip() for b in part.split(":")]
            if len(bits) < 2 or not bits[0] or not bits[1]:
                print(f"[api-keys] skipping malformed entry: {part[:40]!r}")
                continue
            name, secret = bits[0], bits[1]
            limit = 0
            if len(bits) >= 3 and bits[2].isdigit():
                limit = int(bits[2])
            if secret in keys:
                print(f"[api-keys] duplicate secret for {name}; first definition wins")
                continue
            keys[secret] = {"name": name, "limit": limit}
        path = env.get("API_USAGE_FILE") or "/workspace/api-usage.json"
        return cls(keys, path)

    @property
    def enabled(self) -> bool:
        return bool(self._by_secret)

    def limit_for(self, name: str) -> int:
        for meta in self._by_secret.values():
            if meta["name"] == name:
                return meta["limit"]
        return 0

    # ------------------------------------------------------------- persist
    def _load(self) -> None:
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("days"), dict):
                self._days = data["days"]
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[api-keys] state unreadable, starting fresh: {e}")

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
            bucket = self._bucket(meta["name"])
            limit = meta["limit"]
            if limit and bucket["requests"] >= limit:
                raise KeyAuthError(
                    429, "daily_limit_exceeded",
                    f"Daily limit of {limit} requests reached for this key. "
                    f"Resets {_tomorrow()} (UTC).")
        return meta["name"]

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
            bucket = self._bucket(meta["name"])
            if limit and bucket["requests"] >= limit:
                raise KeyAuthError(
                    429, "daily_limit_exceeded",
                    f"Daily limit of {limit} requests reached for this key. "
                    f"Resets {_tomorrow()} (UTC).")
            bucket["requests"] += 1
            self._save()
        return meta["name"]

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
