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
from code_keywords import keyword_queries
from corpus_index import norm
from normalize_query import normalize_query
from situation_prompt import mostly_cyrillic

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

# Issue-level templates: one search per legal issue, never a paraphrase of the
# story and never an article number. Flood+lease must search ijara and zarar
# separately so Suv kodeksi cannot occupy the whole window.
def _phrase_re(phrase):
    words = norm(phrase).split()
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(r"(?<!\w)" + body + r"\w*")


ISSUE_TEMPLATES = [
    {"when": [_phrase_re(p) for p in ("ijara", "ijaraga", "ijarachi", "arend")],
     "query": "ijara shartnomasi Fuqarolik kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "suv bosdi", "toshqin", "suv ostida", "uyimni suv", "zatopil", "potop",
        "zalil", "zalila",
    )],
     "query": "mulkka yetkazilgan zarar qoplash Fuqarolik kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "oylik", "oylk", "maosh", "ish haqi", "ish haqqi", "zarplat", "oklad", "zp",
    )],
     "query": "ish haqini toʻlash muddatlari Mehnat kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "ishdan hayda", "ishdan boʻshat", "ishdan boshat", "ishdan chiqar",
        "uvolil", "uvolen",
    )],
     "query": "mehnat shartnomasini bekor qilish Mehnat kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "armiya", "armiga", "soldat", "harbiy", "chaqiruv", "chaqiriq",
        "mudofaa", "srochka", "povestka", "askar", "dezertir",
    )],
     "query": "Harbiy yoki muqobil xizmatdan boʻyin tovlash muddatli harbiy xizmatga chaqiruv Jinoyat kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "urishib", "urish", "urib", "urdim", "qamash", "qamoq", "kaltak",
        "janjal", "haqorat", "turtib",
    )],
     "query": "yengil tanaga shikast yetkazish kaltaklash haqorat Maʼmuriy javobgarlik kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "ogirlab", "ogirlash", "ogirlik", "oʻgʻirlik", "oʻgʻrilik",
    )],
     "query": "oʻgʻrilik oʻzganing mol-mulkini yashirin ravishda talon-toroj qilish Jinoyat kodeksi"},
    {"when": [_phrase_re(p) for p in ("nalog", "soliq toʻlama", "soliq tolama")],
     "query": "soliq toʻlamaganlik uchun javobgarlik Soliq kodeksi"},
    {"when": [_phrase_re(p) for p in ("aliment", "alimet")],
     "query": "aliment undirish voyaga yetmagan bolalarni taʼminlash Oila kodeksi"},
    {"when": [_phrase_re(p) for p in (
        "pul toʻlamadim", "pul tolamadim", "pul toʻlamasa",
    )],
     "query": "majburiyatni bajarish qarz shartnomasi Fuqarolik kodeksi"},
]


def issue_queries(question):
    """<1s: one short issue+family query per matching template."""
    q = norm(normalize_query(question))
    return [t["query"] for t in ISSUE_TEMPLATES if any(p.search(q) for p in t["when"])]


def pack_lost_in_middle(articles):
    """Best first, second-best last. Models under-attend the middle of a list."""
    arts = list(articles)
    if len(arts) < 2:
        return arts
    return [arts[0], *arts[2:], arts[1]]


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


def exclude_unmentioned_soliq(keys, questions):
    """Drop Soliq hits unless a search query actually names tax.

    Wage/dismissal stories share 'haq toʻlash' with Soliq 371/374 (labour
    income for PIT). Those articles are not labour remedies. Nalog questions
    already search 'Soliq kodeksi' and keep the family.
    """
    blob = norm(" ".join(str(q) for q in questions if q))
    if "soliq" in blob:
        return list(keys)
    return [k for k in keys if k[0] != "soliq_kodeksi"]


def cap_per_code(keys, cap=None, limit=None, prefer=None):
    """Keep at most `cap` articles from each code slug in the final `limit`.

    `prefer` is an optional set of keys (typically title-phrase matches). If a
    preferred key would be dropped because neighbours from the same code already
    filled the cap — JK 166/164 crowding out 169 Oʻgʻrilik — swap the lowest
    non-preferred occupant of that code. Does not invent ranks; RRF order of
    everyone else is unchanged.
    """
    cap = C.PER_CODE_CAP if cap is None else cap
    limit = C.TOP_K if limit is None else limit
    prefer = {tuple(k) for k in (prefer or [])}
    seen, out, slots = {}, [], {}
    for key in keys:
        key_t = tuple(key)
        code = key[0]
        n = seen.get(code, 0)
        if n < cap and len(out) < limit:
            slots.setdefault(code, []).append(len(out))
            seen[code] = n + 1
            out.append(key)
            continue
        if key_t not in prefer:
            continue
        victim = None
        for i in reversed(slots.get(code, [])):
            if tuple(out[i]) not in prefer:
                victim = i
                break
        if victim is None:
            continue
        out[victim] = key
    return out


def queries_for(question, expander, complete_fn=None):
    """Normalize, then original+expand first, then issue templates.

    27B only if still under 3 queries. Retrieval queries stay Latin-Uzbek:
    a Cyrillic story is transliterated before BM25 against a Latin corpus.
    Issue templates (and the rewrite) already emit Latin.
    """
    question = normalize_query(question)
    extras = issue_queries(question) + keyword_queries(question)
    if expander is not None and hasattr(expander, "issue_adds"):
        extras = extras + expander.issue_adds(question)
    if mostly_cyrillic(question):
        first = ""
    else:
        first = expander.expand(question) if expander is not None else question
    merged = merge_queries(first, extras)
    if len(merged) >= 3 or complete_fn is None:
        return merged
    try:
        rewritten = parse_query_list(complete_fn(question) or "")
    except Exception:
        rewritten = []
    return merge_queries(first, extras + rewritten)
