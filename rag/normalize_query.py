"""Query-side spelling / dialect / SMS normalization.

Retrieval is already hybrid over full article text (bge-m3 + BM25 + RRF).
Everyday Uzbek and Russian chat does not use statute tokens (`armiya` vs
`harbiy xizmat`, `oylk` vs `oylik`), so BM25 greps the wrong string. This
module rewrites the *query* into Latin-Uzbek search language. It does not
invent article numbers and does not touch the indexed corpus.

Applied before synonym expand + retrieve. Deterministic `N-modda` lookup
keeps working: digit SMS maps are skipped when the string looks like a
citation (`999-modda`, `141²`, `JK 253-modda`), and counts glued to digits
(`4ta`, `6oy`) are never read as letter-play. Telegram latin (`w` for `sh`,
`c` for `ch`) is the real customer path. Reform letters (`ş ç ō ğ ñ`) are
almost unused; they still map, so a rare query does not miss, but they are
not a traffic cluster. English/URL tokens are parked first so `whatsapp`
and `camera` stay intact.
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

# A digit glued to a numeral suffix is a count, not letter-play: "4ta bola",
# "6oy", "4kishi", "2 yil 6oy". Substituting there turns the count into a
# garbage token ("chta") or a different real word ("oʻta").
_NUM_SUFFIX = (
    "tasini|tadan|tasi|tada|ta|"
    "oylari|oylik|oyda|oyi|oy|"
    "yilda|yili|yil|"
    "kishidan|kishi|"
    "martadan|marta|nchisi|nchi|chi|"
    "xil|dona|foiz|protsent|"
    "mln|mlrd|ming|yuz|km|kg|sm|mm|soʻm|"
    "dan|da|ga|ni|sinf|kurs|bet|qavat|xona"
)

# Isolated 6/4 glued to letters: t6lov, 6zbek, 4iqarish. Not 253, not 2 oy,
# not a count (4ta -- suffix guard), not a product code (A4, 4x4 -- the digit
# must have letters on both sides and no second digit later in the token).
_SMS_GUARD = r"(?!(?:" + _NUM_SUFFIX + r")|[A-Za-z\u02bb\u02bc]*\d)"
_SMS_6 = re.compile(
    r"(?:(?<=[A-Za-z\u02bb\u02bc])6(?=[A-Za-z\u02bb\u02bc])" + _SMS_GUARD +
    r"|(?<![0-9A-Za-z\u02bb\u02bc])6(?=[A-Za-z\u02bb\u02bc])" + _SMS_GUARD + r")",
    re.IGNORECASE,
)
_SMS_4 = re.compile(
    r"(?:(?<=[A-Za-z\u02bb\u02bc])4(?=[A-Za-z\u02bb\u02bc])" + _SMS_GUARD +
    r"|(?<![0-9A-Za-z\u02bb\u02bc])4(?=[A-Za-z\u02bb\u02bc])" + _SMS_GUARD + r")",
    re.IGNORECASE,
)

# www is a URL token, not excitement: _TRIPLE_LETTER must not collapse it.
_WWW = re.compile(r"\bwww\b", re.IGNORECASE)

# Rare reform / Turkish-keyboard letters. Almost nobody types these in
# Telegram; keep the map so a stray query still hits the corpus.
_NEW_ALPHA = {
    "ş": "sh", "Ş": "Sh",
    "ç": "ch", "Ç": "Ch",
    "ō": "o" + OKINA, "Ō": "O" + OKINA,
    "ğ": "g" + OKINA, "Ğ": "G" + OKINA,
    "ḡ": "g" + OKINA, "Ḡ": "G" + OKINA,
    "ñ": "ng", "Ñ": "Ng",
    "ı": "i", "İ": "I",
}

# Whole-token English / brand / TLD holds. A blanket w→sh or c→ch would
# smash whatsapp, camera, card, power, new, kiwi — mixed into Uzbek chat.
_ENGLISH_TOKENS = (
    "whatsapp", "windows", "window", "workflow", "wikipedia", "website",
    "wifi", "wiki", "web", "www", "power", "wow", "wait", "want", "was",
    "were", "with", "will", "would", "what", "when", "where", "who", "why",
    "which", "we", "week", "work", "world", "water", "wallet", "how", "now",
    "low", "know", "down", "show", "own", "owner", "law", "lawyer", "new",
    "news", "two", "twelve", "twenty", "twitter", "twitch", "review", "view",
    "follow", "yellow", "slow", "grow", "brown", "town", "draw", "raw", "saw",
    "kiwi", "answer", "between", "always", "forward", "network", "password",
    "software", "hardware", "download", "huawei",
    "facebook", "instagram", "camera", "cameras", "card", "cards", "code",
    "codes", "court", "case", "cases", "company", "customer", "contract",
    "contracts", "document", "documents", "office", "service", "license",
    "police", "price", "once", "percent", "official", "public", "republic",
    "face", "nice", "scam", "scan", "fact", "act", "credit", "account",
    "com", "google", "youtube", "apple", "samsung", "iphone", "ipad",
    "telegram", "online", "bank",
)
_ENGLISH_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(w) for w in sorted(_ENGLISH_TOKENS, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|\b[a-z0-9][a-z0-9-]*(?:\.[a-z]{2,})+\b",
    re.IGNORECASE,
)

# c→ch only when it is standing in for Uzbek ch (koca, qanca, haydamoqci).
# Skip sc/cc, already-ch, and English clusters (ct, ck, cl, cr).
_C_TO_CH = re.compile(
    r"(?<![sScC])[cC](?![hH])(?=[aeiouyAEIOUY\u02bb\u02bc]|\W|$)"
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
    # "armiya" has zero postings in the corpus; "harbiy xizmat" has ~200.
    # The expander/keyword layers all trigger on "harbiy xizmat" too.
    ("armiga", "harbiy xizmat"),
    ("ogirlab", "oʻgʻrilik"),
    ("ogirlash", "oʻgʻrilik"),
    ("oʻgʻirlik", "oʻgʻrilik"),
    ("ogirlik", "oʻgʻrilik"),
    ("armiyu", "harbiy xizmat"),
    ("armii", "harbiy xizmat"),
    ("armiey", "harbiy xizmat"),
    ("armiyey", "harbiy xizmat"),
    ("armiya", "harbiy xizmat"),
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


def _new_alphabet(text: str) -> str:
    """ş/ç/ō/ğ/ñ/ı → corpus Latin. Rare input; cheap and collision-free."""
    return "".join(_NEW_ALPHA.get(ch, ch) for ch in text)


def _protect_latin(text: str) -> tuple[str, list[str]]:
    """Park URLs and English tokens so w→sh / c→ch cannot touch them."""
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x01{len(held) - 1}\x01"

    text = _URL_RE.sub(hold, text)
    text = _ENGLISH_RE.sub(hold, text)
    return text, held


def _restore_latin(text: str, held: list[str]) -> str:
    for i, orig in enumerate(held):
        text = text.replace(f"\x01{i}\x01", orig)
    return text


def _sms_w(text: str) -> str:
    """Remaining w is Telegram-sh (iwxona, ketiwdi, qamawadimi, wartnoma).

    English / URL tokens must already be parked by `_protect_latin`. A
    previous wu-only map left `iw` and `aw` looking like they never said ish.
    """
    def repl(m: re.Match) -> str:
        return "Sh" if m.group(0).isupper() else "sh"

    return re.sub(r"[wW]", repl, text)


def _sms_c(text: str) -> str:
    """Remaining c before a vowel (or end) is Telegram-ch (koca, qanca, moqci)."""
    def repl(m: re.Match) -> str:
        return "Ch" if m.group(0).isupper() else "ch"

    return _C_TO_CH.sub(repl, text)


def _collapse_repeats(text: str) -> str:
    """3+ same letters → one, but never inside www (it is a URL, not excitement)."""
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{len(held) - 1}\x00"

    text = _WWW.sub(hold, text)
    text = _TRIPLE_LETTER.sub(r"\1", text)
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
    # Reform letters before w/c so iş/çiqarish become ish/chiqarish in one pass.
    text = _new_alphabet(text)
    # Park English/URLs, then map leftover w→sh and c→ch, including next to
    # a citation: "JK 169-modda ketiwdi" must keep 169 and still become ketishdi.
    text, held = _protect_latin(text)
    text = _sms_w(text)
    text = _sms_c(text)
    text = _restore_latin(text, held)
    text = _collapse_repeats(text)
    text = _apply_words(text)
    for pat, dst in _GLUED:
        text = pat.sub(dst, text)
    return re.sub(r"\s+", " ", text).strip()
