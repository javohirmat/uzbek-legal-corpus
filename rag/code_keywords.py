"""Colloquial labels for the 25 local codes — a retrieval *boost*, not a gate.

People say "armiya", "havo", "asosiy qonun"; the statute says "harbiy xizmat",
"samoviy hudud", "Konstitutsiya". After normalize_query, match those everyday
tokens and add one extra search string of statute-ish phrases for that code.
Hybrid bge-m3+BM25 stays the path. Missing a synonym must not zero the search.

Does not invent article numbers. Does not rebuild the index.
"""
from __future__ import annotations

import json
import os
import re

import config as C
from corpus_index import norm
from normalize_query import normalize_query


def _phrase_re(phrase):
    words = norm(phrase).split()
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(r"(?<!\w)" + body + r"\w*")


def pretty_code(slug):
    logical = re.sub(r"[_\-]\d+qism$", "", slug or "")
    return " ".join(re.split(r"[_\-]+", logical)).capitalize()


def cap_for_matches(codes, cap=None, only_cap=None):
    """One matched code may fill the window; several keep the per-code cap."""
    cap = C.PER_CODE_CAP if cap is None else cap
    only_cap = C.TOP_K if only_cap is None else only_cap
    n = len({c for c in (codes or []) if c})
    if n == 1:
        return only_cap
    return cap


def boost_matched_codes(keys, codes):
    """Already-retrieved keys from matched codes rise. Nothing is invented."""
    want = {c for c in (codes or []) if c}
    if not want:
        return list(keys)
    boosted = [k for k in keys if k[0] in want]
    rest = [k for k in keys if k[0] not in want]
    return boosted + rest


class CodeKeywords:
    def __init__(self, entries):
        self.entries = []
        for raw in entries or []:
            code = (raw.get("code") or "").strip()
            if not code:
                continue
            phrases = []
            for field in ("informal_uz", "informal_ru", "sms"):
                phrases.extend(raw.get(field) or [])
            pats = []
            for phrase in phrases:
                rx = _phrase_re(phrase)
                if rx is not None:
                    pats.append(rx)
            formal = [p for p in (raw.get("formal") or []) if str(p).strip()]
            if pats:
                self.entries.append({
                    "code": code,
                    "pats": pats,
                    "formal": formal,
                    "pretty": pretty_code(code),
                })

    @classmethod
    def load(cls, path=None):
        path = path or getattr(C, "CODE_KEYWORDS_JSON", "")
        if not path or not os.path.exists(path):
            return cls([])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("codes") or data.get("entries") or []
        return cls(data)

    def match(self, question):
        """Codes whose colloquial labels hit the *normalized* query."""
        q = norm(normalize_query(question or ""))
        if not q:
            return []
        hits = []
        seen = set()
        for e in self.entries:
            if e["code"] in seen:
                continue
            if any(p.search(q) for p in e["pats"]):
                seen.add(e["code"])
                hits.append(e)
        return hits

    def extra_queries(self, question):
        """One statute-ish search string per matched code. No article numbers."""
        out = []
        for e in self.match(question):
            body = " ".join(e["formal"]).strip()
            q = f"{body} {e['pretty']}".strip() if body else e["pretty"]
            if q:
                out.append(q)
        return out

    def slugs(self, question):
        return [e["code"] for e in self.match(question)]

    def __len__(self):
        return len(self.entries)


_catalog = None


def get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = CodeKeywords.load()
    return _catalog


def match_codes(question):
    return get_catalog().match(question)


def match_code_slugs(question):
    return get_catalog().slugs(question)


def keyword_queries(question):
    return get_catalog().extra_queries(question)
