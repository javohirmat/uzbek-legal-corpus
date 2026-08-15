#!/usr/bin/env python3
"""Retrieval-only harness for situational gold.

Does not import pipeline.py (that would spin up vLLM). Three backends:

  python eval_recall.py --dry-run
      Validate every gold id against local JSONL (fallback: data/raw headings).
      No GPU, no network, no Chroma.

  python eval_recall.py --url http://140.150.159.1:22879
      POST /retrieve (or /v1/retrieve, /search) on a live RAG box.

  python eval_recall.py
      Import Retriever if Chroma is on disk; otherwise a CPU lexical (BM25 /
      token-overlap) fallback so the script still runs without GPU.

Prints recall@K and wrong-code rate. Exit 1 if any gold id is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

GOLD_DEFAULT = HERE / "eval" / "gold_situations.jsonl"
ARTICLES_DEFAULT = REPO / "data" / "articles"
RAW_DEFAULT = REPO / "data" / "raw"

_ART_IN_QUERY = re.compile(r"\d+\s*[-‐‑‒–—]?\s*modda", re.IGNORECASE)
_HEAD = re.compile(r"^\s*(\d+)\s*-\s*modda\b\.?\s*(.*)$")
_CITABLE = re.compile(r"^\d+(\.\d+)?$")


def _key(code: str, article_id: str) -> str:
    return f"{code}::{article_id}"


def parse_gold_id(raw: str) -> tuple[str, str]:
    if "::" in raw:
        code, aid = raw.split("::", 1)
    elif ":" in raw:
        code, aid = raw.split(":", 1)
    else:
        raise ValueError(f"gold id must be code::article_id, got {raw!r}")
    return code.strip(), aid.strip()


def resolve_ids(nums: list[int]) -> list[tuple[str, str]]:
    """Same position-aware walk as scripts/parse_articles.py (do not import it:
    that module runs the full rewrite of data/articles on import)."""
    out, prev = [], (nums[0] - 1 if nums and nums[0] > 1 else 0)
    for j, n in enumerate(nums):
        nxt = nums[j + 1] if j + 1 < len(nums) else None
        cand = None
        for length in range(1, len(str(n))):
            b = int(str(n)[:length])
            if prev <= b <= prev + 40:
                cand = (b, str(n)[length:])
                break
        base_ok = prev < n <= prev + 40
        if base_ok and cand and cand[0] != n:
            if nxt is not None and (nxt == cand[0] + 1 or str(nxt).startswith(str(cand[0]))):
                out.append((f"{cand[0]}.{int(cand[1])}", "insert"))
                prev = max(prev, cand[0])
                continue
            out.append((str(n), "base"))
            prev = n
            continue
        if base_ok:
            out.append((str(n), "base"))
            prev = n
            continue
        if cand:
            out.append((f"{cand[0]}.{int(cand[1])}", "insert"))
            prev = max(prev, cand[0])
            continue
        out.append((str(n), "unresolved"))
    return out


def load_jsonl_index(articles_dir: Path) -> tuple[set[str], list[dict]]:
    ids, records = set(), []
    if not articles_dir.is_dir():
        return ids, records
    for path in sorted(articles_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                code, aid = row.get("code") or "", row.get("article_id") or ""
                if not code or not aid:
                    continue
                rec = {
                    "code": code,
                    "article_id": aid,
                    "code_title": row.get("code_title") or "",
                    "article_display": row.get("article_display") or "",
                    "title": row.get("title") or "",
                    "text": row.get("text") or "",
                }
                records.append(rec)
                if _CITABLE.fullmatch(aid):
                    ids.add(_key(code, aid))
    return ids, records


def load_raw_index(raw_dir: Path) -> set[str]:
    ids = set()
    if not raw_dir.is_dir():
        return ids
    for path in sorted(raw_dir.glob("*.txt")):
        code = path.stem
        nums = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _HEAD.match(line)
            if m:
                nums.append(int(m.group(1)))
        for aid, _kind in resolve_ids(nums):
            if _CITABLE.fullmatch(aid):
                ids.add(_key(code, aid))
    return ids


def load_gold(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("id", "lang", "query", "gold_article_ids", "wrong_codes", "notes"):
                if field not in row:
                    raise SystemExit(f"{path}:{i} missing field {field!r}")
            if row["lang"] not in ("uz", "ru"):
                raise SystemExit(f"{path}:{i} lang must be uz|ru, got {row['lang']!r}")
            gold = row["gold_article_ids"]
            if not isinstance(gold, list) or not 1 <= len(gold) <= 4:
                raise SystemExit(f"{path}:{i} gold_article_ids must have 1–4 ids")
            row["_parsed"] = [parse_gold_id(g) for g in gold]
            rows.append(row)
    return rows


def validate_gold(rows: list[dict], corpus_ids: set[str]) -> list[str]:
    errors = []
    if not corpus_ids:
        errors.append("no local corpus ids (data/articles/*.jsonl and data/raw/*.txt empty)")
        return errors
    for row in rows:
        q = row["query"]
        if _ART_IN_QUERY.search(q):
            errors.append(f"{row['id']}: query names an article number")
        for raw, (code, aid) in zip(row["gold_article_ids"], row["_parsed"]):
            key = _key(code, aid)
            if key not in corpus_ids:
                errors.append(f"{row['id']}: gold id not in corpus: {raw}")
        for code in row["wrong_codes"]:
            if any(key.startswith(code + "::") for key in corpus_ids):
                continue
            # slug may still be valid even if this checkout only has a subset
            if not any(key.split("::", 1)[0] == code for key in corpus_ids):
                errors.append(f"{row['id']}: wrong_codes slug not a known code: {code}")
    return errors


# ----- retrieval backends -------------------------------------------------


def _normalize_hit(item) -> tuple[str, str] | None:
    if item is None:
        return None
    if isinstance(item, str):
        try:
            return parse_gold_id(item)
        except ValueError:
            return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    if isinstance(item, dict):
        code = item.get("code") or item.get("slug")
        aid = item.get("article_id") or item.get("article") or item.get("id")
        if isinstance(aid, str) and "-modda" in aid.lower():
            aid = re.sub(r"[^\d.]", "", aid.replace("¹", ".1").replace("³", ".3"))
        if code and aid:
            if "::" in str(code):
                return parse_gold_id(str(code))
            return str(code), str(aid)
    return None


def hits_from_payload(payload, k: int) -> list[tuple[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("keys")
            or payload.get("results")
            or payload.get("hits")
            or payload.get("articles")
            or payload.get("citations")
            or payload.get("ids")
            or []
        )
    else:
        items = []
    out, seen = [], set()
    for item in items:
        hit = _normalize_hit(item)
        if not hit or hit in seen:
            continue
        seen.add(hit)
        out.append(hit)
        if len(out) >= k:
            break
    return out


def retrieve_http(base: str, query: str, k: int) -> list[tuple[str, str]]:
    base = base.rstrip("/")
    body = json.dumps({"query": query, "question": query, "k": k}).encode("utf-8")
    paths = ("/retrieve", "/v1/retrieve", "/search")
    last_err = None
    for path in paths:
        req = urllib.request.Request(
            base + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            hits = hits_from_payload(payload, k)
            if hits or resp.status == 200:
                return hits
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404:
                continue
            raise SystemExit(f"HTTP {e.code} {base + path}: {e.read()[:200]!r}") from e
        except urllib.error.URLError as e:
            last_err = e
            continue
    raise SystemExit(
        f"no retrieve endpoint at {base} (tried {', '.join(paths)}). "
        f"Last error: {last_err}. Pass --dry-run to validate gold only, "
        "or run against a box that exposes POST /retrieve. "
        "Will not call /v1/chat/completions (that is generation, not retrieval)."
    )


class LexicalRetriever:
    """CPU-only fallback. BM25 if rank_bm25 is installed, else token overlap."""

    def __init__(self, records: list[dict]):
        self.records = [r for r in records if _CITABLE.fullmatch(r["article_id"])]
        self._docs = [
            re.findall(
                r"\w+",
                f'{r["code"]} {r["code_title"]} {r["article_display"]} {r["title"]} {r["text"]}'.lower(),
            )
            for r in self.records
        ]
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._docs)
        except Exception:
            pass

    def search(self, question: str, k: int = 6):
        q = re.findall(r"\w+", question.lower())
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            keys = [(self.records[i]["code"], self.records[i]["article_id"]) for i in order]
            return keys, 0.0
        qset = set(q)
        scored = []
        for rec, toks in zip(self.records, self._docs):
            overlap = len(qset & set(toks))
            if overlap:
                scored.append((overlap, rec["code"], rec["article_id"]))
        scored.sort(reverse=True)
        return [(c, a) for _, c, a in scored[:k]], 0.0


def retrieve_local(records: list[dict], query: str, k: int) -> tuple[list[tuple[str, str]], str]:
    try:
        import config as C
        from retriever import Retriever
    except Exception as e:
        lex = LexicalRetriever(records)
        keys, _ = lex.search(query, k=k)
        return keys, f"offline-lexical ({type(e).__name__}: no live Retriever)"

    chroma = Path(C.CHROMA_DIR)
    if not chroma.exists():
        lex = LexicalRetriever(records)
        keys, _ = lex.search(query, k=k)
        return keys, "offline-lexical (no chroma/ index)"

    arts = [r for r in records if _CITABLE.fullmatch(r["article_id"])]
    try:
        retr = retrieve_local._cached
    except AttributeError:
        retr = Retriever(arts)
        retrieve_local._cached = retr
    if hasattr(retr, "search_multi"):
        keys, _ = retr.search_multi([query], k=k)
    else:
        keys, _ = retr.search(query, k=k)
    return [(c, a) for c, a in keys[:k]], "local-retriever"


# ----- metrics ------------------------------------------------------------


def eval_row(row: dict, hits: list[tuple[str, str]]) -> dict:
    gold = set(row["_parsed"])
    hit_set = set(hits)
    recalled = gold & hit_set
    recall = len(recalled) / len(gold) if gold else 0.0
    gold_codes = {c for c, _ in gold}
    wrong = [c for c, _ in hits if c in row["wrong_codes"] and c not in gold_codes]
    return {
        "id": row["id"],
        "recall": recall,
        "hits": len(recalled),
        "gold": len(gold),
        "wrong_code_hits": len(wrong),
        "retrieved": [_key(c, a) for c, a in hits],
        "recalled": [_key(c, a) for c, a in recalled],
        "wrong": wrong,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    p.add_argument("--articles", type=Path, default=Path(
        os.getenv("LOCAL_ARTICLES", str(ARTICLES_DEFAULT))
    ))
    p.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--dry-run", action="store_true",
                   help="only validate gold ids against local JSONL/raw")
    p.add_argument("--url", default="",
                   help="RAG base URL, e.g. http://140.150.159.1:22879")
    p.add_argument("--json", action="store_true", help="print per-row JSON")
    args = p.parse_args(argv)

    rows = load_gold(args.gold)
    corpus_ids, records = load_jsonl_index(args.articles)
    source = f"jsonl:{args.articles} ({len(corpus_ids)} ids)"
    if not corpus_ids:
        corpus_ids = load_raw_index(args.raw)
        source = f"raw:{args.raw} ({len(corpus_ids)} ids)"
        records = []

    errors = validate_gold(rows, corpus_ids)
    print(f"gold={len(rows)}  corpus={source}")
    if errors:
        print(f"FAIL  {len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS  every gold id exists; queries name no article numbers")

    if args.dry_run:
        langs = Counter(r["lang"] for r in rows)
        print(f"dry-run ok  langs={dict(langs)}")
        return 0

    backend = "http" if args.url else "local"
    reports = []
    backend_label = backend
    for row in rows:
        if args.url:
            hits = retrieve_http(args.url, row["query"], args.k)
            backend_label = f"http:{args.url}"
        else:
            hits, backend_label = retrieve_local(records, row["query"], args.k)
        reports.append(eval_row(row, hits))

    mean_recall = sum(r["recall"] for r in reports) / len(reports)
    wrong_rows = sum(1 for r in reports if r["wrong_code_hits"])
    wrong_rate = wrong_rows / len(reports)
    print(f"backend={backend_label}  k={args.k}")
    print(f"recall@{args.k}={mean_recall:.3f}  wrong-code rate={wrong_rate:.3f}  "
          f"({wrong_rows}/{len(reports)} rows pulled a listed trap code)")
    for r in reports:
        flag = "W" if r["wrong_code_hits"] else " "
        print(f"  {flag} {r['id']:22}  recall={r['recall']:.2f}  "
              f"{r['hits']}/{r['gold']}  retrieved={r['retrieved'][:args.k]}")
        if args.json:
            print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
