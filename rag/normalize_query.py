"""Query-side spelling / dialect / SMS normalization.

Retrieval is already hybrid over full article text (bge-m3 + BM25 + RRF).
Everyday Uzbek and Russian chat does not use statute tokens (`armiya` vs
`harbiy xizmat`, `oylk` vs `oylik`), so BM25 greps the wrong string. This
module rewrites the *query* into Latin-Uzbek search language. It does not
invent article numbers and does not touch the indexed corpus.

Applied before synonym expand + retrieve. Deterministic `N-modda` lookup
keeps working: digit SMS maps are skipped when the string looks like a
citation (`999-modda`, `141²`, `JK 253-modda`).
"""
from __future__ import annotations

import re
import unicodedata

# Corpus encoding: okina in o/g is U+02BB; tutuq belgisi is U+02BC.
OKINA = "\u02bb"   # ʻ
TUTUQ = "\u02bc"   # ʼ

# Apostrophe-like characters people actually type (ASCII, curly, modifier).
_APOS_CHARS = (
    "'\u2018\u2019\u201b\u2032\u02b9\u02bb\u02bc\u02bd"
    "\u02bc\u02bb`´ʹʺ‛"
)

# 6 / 4 as letter-substitutes must never rewrite a citation or a money amount.
_CITATION = re.compile(
    r"(?:"
    r"\d+\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]*\s*[-‐‑‒–—]?\s*(?:modda|модда)"
    r"|\b(?:jk|mk|fk|jpk|mjk|fpk|ipk)\b"
    r")",
    re.IGNORECASE,
)

# Isolated 6/4 glued to letters: t6lov, 6zbek, 4iqarish. Not 253, not 2 oy.
_SMS_6 = re.compile(
    r"(?:(?<=[A-Za-z\u02bb\u02bc])6(?!\d)|(?<!\d)6(?=[A-Za-z\u02bb\u02bc]))"
)
_SMS_4 = re.compile(
    r"(?:(?<=[A-Za-z\u02bb\u02bc])4(?!\d)|(?<!\d)4(?=[A-Za-z\u02bb\u02bc]))"
)

_EN_W = re.compile(
    r"\b(?:www|https?|what|who|where|when|why|how|will|was|were|with|"
    r"we|would|which|whose|tweet|twitter)\b",
    re.IGNORECASE,
)

# 3+ of the same letter → one. Never digits (999-modda) or punctuation.
_TRIPLE_LETTER = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)

_GLUED = (
    (re.compile(r"nimaboladi", re.IGNORECASE), "nima boladi"),
    (re.compile(r"nimaqilishadi", re.IGNORECASE), "nima qilishadi"),
    (re.compile(r"ishdanhayda", re.IGNORECASE), "ishdan hayda"),
    (re.compile(r"armiyagabor", re.IGNORECASE), "armiyaga bor"),
    (re.compile(r"harbiyxizmat", re.IGNORECASE), "harbiy xizmat"),
)

# After transliteration. Longer keys first. Whole-token only.
_WORDS = (
    ("nma", "nima"),
    ("nm", "nima"),
    ("oylk", "oylik"),
    ("oylig", "oylik"),
    ("oylg", "oylik"),
    ("zp", "oylik"),
    ("zarplata", "oylik"),
    ("zarplatu", "oylik"),
    ("zarplaty", "oylik"),
    ("zarplati", "oylik"),
    ("oklad", "oylik"),
    ("haydadim", "haydash"),
    ("haydadi", "haydash"),
    ("haydad", "haydash"),
    ("uvolili", "ishdan boʻshatish"),
    ("uvolil", "ishdan boʻshatish"),
    ("uvolnenie", "ishdan boʻshatish"),
    ("uvolnen", "ishdan boʻshatish"),
    ("alimenty", "aliment"),
    ("alimenti", "aliment"),
    ("alimentov", "aliment"),
    ("srochka", "muddatli harbiy xizmat"),
    ("srochku", "muddatli harbiy xizmat"),
    ("srochki", "muddatli harbiy xizmat"),
    ("povestka", "chaqiruv"),
    ("povestku", "chaqiruv"),
    ("povestki", "chaqiruv"),
    ("armiga", "armiya"),
    ("ogirlab", "oʻgʻrilik"),
    ("ogirlash", "oʻgʻrilik"),
    ("oʻgʻirlik", "oʻgʻrilik"),
    ("ogirlik", "oʻgʻrilik"),
    ("armiyu", "armiya"),
    ("armii", "armiya"),
    ("armiey", "armiya"),
    ("armiyey", "armiya"),
    ("soldata", "soldat"),
    ("soldatu", "soldat"),
    ("soldaty", "soldat"),
    ("dezertirstvo", "dezertirlik"),
    ("dezertir", "dezertirlik"),
)

_WORD_RE = [
    (re.compile(r"(?<!\w)" + re.escape(src) + r"(?!\w)", re.IGNORECASE), dst)
    for src, dst in _WORDS
]

# Single Cyrillic codepoints → Latin-Uzbek (official okinas). Digraphs are
# still one Cyrillic letter.
_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sh", "ъ": TUTUQ, "ы": "i", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + OKINA, "қ": "q", "ғ": "g" + OKINA, "ҳ": "h",
    "ї": "i", "і": "i", "ґ": "g",
}


def looks_like_citation(text: str) -> bool:
    """True when digit SMS maps would corrupt an article reference."""
    return bool(text) and _CITATION.search(text) is not None


def _canon_apostrophes(text: str) -> str:
    """o'/g' variants → okina; other apostrophes → tutuq. Matches corpus."""
    apos = set(_APOS_CHARS)
    out = []
    for ch in text:
        if ch in apos:
            prev = out[-1] if out else ""
            out.append(OKINA if prev.lower() in ("o", "g") else TUTUQ)
        else:
            out.append(ch)
    return "".join(out)


def _map_cyr_char(ch: str) -> str:
    lat = _CYR.get(ch.lower())
    if lat is None:
        return ch
    if ch.isupper() and lat:
        return lat[0].upper() + lat[1:]
    return lat


def transliterate_cyrillic(text: str) -> str:
    """Uzbek + Russian Cyrillic → Latin-Uzbek. Latin letters unchanged."""
    return "".join(_map_cyr_char(ch) for ch in text)


def _sms_w(text: str) -> str:
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{len(held) - 1}\x00"

    text = _EN_W.sub(hold, text)

    def repl(m: re.Match) -> str:
        return "Sh" if m.group(0).isupper() else "sh"

    text = re.sub(r"w", repl, text, flags=re.IGNORECASE)
    for i, orig in enumerate(held):
        text = text.replace(f"\x00{i}\x00", orig)
    return text


def _apply_words(text: str) -> str:
    for pat, dst in _WORD_RE:
        text = pat.sub(dst, text)
    return text


def normalize_query(text: str) -> str:
    """Rewrite messy user input into Latin-Uzbek search text.

    Idempotent enough to run at more than one call site. Preserves `JK`,
    `999-modda`, and superscript article ids.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _canon_apostrophes(text)
    skip_digits = looks_like_citation(text)
    text = transliterate_cyrillic(text)
    if not skip_digits:
        text = _SMS_6.sub("o" + OKINA, text)
        text = _SMS_4.sub("ch", text)
        text = _sms_w(text)
    text = _TRIPLE_LETTER.sub(r"\1", text)
    text = _apply_words(text)
    for pat, dst in _GLUED:
        text = pat.sub(dst, text)
    return re.sub(r"\s+", " ", text).strip()
