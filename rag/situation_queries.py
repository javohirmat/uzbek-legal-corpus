"""Situational multi-query: a story is not one embedding.

A flood-and-lease narrative embeds nearest to Suv kodeksi; the civil-law
issues (ijara, zarar) never enter the six-article window. Generate 3–5 short
search queries (issue + likely code family), always keep the original+expand
query first, and let the retriever RRF-fuse the lists.

The rewrite call is thinking-off and budgeted. Timeout or garbage JSON falls
back to the original expanded query — never blocks named-article lookup.
"""
import json
import re

import config as C
from corpus_index import norm

REWRITE_SYSTEM = (
    "Sen qonun qidiruv yordamchisisan. Foydalanuvchi vaziyatini 3-5 ta "
    "QISQA lotin-oʻzbek qidiruv soʻroviga aylantir. Har bir soʻrov: huquqiy "
    "masala + ehtimoliy kodeks oilasi (Fuqarolik, Mehnat, Suv, Jinoyat va hokazo). "
    "Hech qachon modda raqamini oʻylab topma va hech qachon 'N-modda' yozma. "
    "Javob FAQAT JSON massiv, boshqa matn yoʻq: "
    "[\"soʻrov 1\", \"soʻrov 2\", \"soʻrov 3\"]"
)

# Rewrite must not invent citations — drop any query that looks like one.
_ART_NUM = re.compile(r"\d+\s*[-‐‑‒–—]?\s*modda", re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def parse_query_list(raw):
    """Best-effort JSON array of short strings. Empty on any garbage."""
    if not raw or not isinstance(raw, str):
        return []
    text = _FENCE.sub("", raw.strip()).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("queries") or data.get("soʻrovlar") or data.get("sorovlar") or []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict):
            q = str(item.get("query") or item.get("q") or "").strip()
        else:
            q = ""
        if q:
            out.append(q)
    return out


def merge_queries(original, rewritten, limit=None):
    """Original (already expanded) is always query #1. Drop dupes and invented articles."""
    limit = C.SITUATION_MAX_QUERIES if limit is None else limit
    first = (original or "").strip()
    out, seen = [], set()
    if first:
        out.append(first)
        seen.add(norm(first))
    for q in rewritten or []:
        q = q.strip()
        if not q or _ART_NUM.search(q):
            continue
        key = norm(q)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def rrf_fuse(ranked_lists, k_const=None):
    """Reciprocal rank fusion across already-ranked key lists. Score-free."""
    k_const = C.RRF_K if k_const is None else k_const
    fused = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k_const + rank)
    return sorted(fused, key=lambda key: -fused[key])


def cap_per_code(keys, cap=None, limit=None):
    """Keep at most `cap` articles from each code slug in the final `limit`."""
    cap = C.PER_CODE_CAP if cap is None else cap
    limit = C.TOP_K if limit is None else limit
    seen, out = {}, []
    for key in keys:
        code = key[0]
        n = seen.get(code, 0)
        if n >= cap:
            continue
        seen[code] = n + 1
        out.append(key)
        if len(out) >= limit:
            break
    return out


def queries_for(question, expander, complete_fn=None):
    """Return 1–5 search strings. complete_fn is optional (tests pass None)."""
    first = expander.expand(question) if expander is not None else question
    rewritten = []
    if complete_fn is not None:
        try:
            rewritten = parse_query_list(complete_fn(question) or "")
        except Exception:
            rewritten = []
    return merge_queries(first, rewritten)
