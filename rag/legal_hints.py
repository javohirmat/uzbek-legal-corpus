"""Legal-vocabulary gate so statute questions are not answered as chat.

Latin stems used to live only as `_LEGAL_HINT` in pipeline.py. Informal
spellings (`armiga`, `qamashadimi`, `povestka`) never contained `modda` or
`armiya`, so the cosine fallback sent them to general chat — which then
invented jarima / qamash from memory. Cyrillic stems are the same gate for
Russian wages/firing stories.

Query normalization (armiga→armiya, povestka→chaqiruv) still runs first in
the retrieve path. This module only decides *whether* to retrieve.
"""
import re

from corpus_index import norm

# Stems, not whole words: зарплат matches зарплата/зарплату, увольн matches
# уволил/уволили/увольнение. Keep this list Cyrillic-only.
CYRILLIC_LEGAL_CUES = (
    "закон",
    "статья",
    "зарплат",
    "уволил",
    "увольн",
    "трудов",
    "договор",
    "штраф",
    "налог",
    "аренд",
    "наслед",
    "развод",
    "алимент",
    "уголовн",
    "гражданск",
    "кодекс",
    "юрист",
    "нотариус",
    "конституц",
    "лиценз",
    "пенси",
    "исков",
    "истец",
    "суд",
    "права",
    "право",
    "обязанност",
    "имуществ",
    "собственн",
    "возмещен",
    "затопил",
    "потоп",
    "работодател",
    "работник",
    "таможен",
    "брак",
    "женит",
    "опек",
    "қонун",
    "модда",
    "ҳуқуқ",
    "жарима",
    "ижара",
    "меҳнат",
    "жиноят",
    "фуқаролик",
    "шартнома",
    "солиқ",
    "никоҳ",
    "мерос",
    "ойлик",
    "иш ҳақ",
    "ишдан ҳайда",
    "ишдан бўшат",
    "арми",
    "солдат",
    "срочк",
    "военн",
    "призыв",
    "повестк",
    "дезертир",
    "милиц",
    "полиц",
    "арест",
    "посад",
    "драк",
    "изби",
    "тюрм",
    "залил",
    "зп",
)

# Everyday Latin / SMS. Informal spellings included so the gate fires even
# when normalize_query has not yet rewritten the string. Do not add bare
# `nma` / `boladi` — those are half of all chat. `urish` needs a leading
# boundary so it does not fire inside `qurilish`.
LATIN_LEGAL_CUES = (
    "modda",
    "kodeks",
    "qonun",
    "huquq",
    "jazo",
    "sud",
    "shartnoma",
    "majburiyat",
    "javobgarlik",
    "jinoyat",
    "fuqarolik",
    "mehnat",
    "soliq",
    "bojxona",
    "ijara",
    "nikoh",
    "meros",
    "farzandlikka",
    "davo",
    "konstitutsiya",
    "litsenziya",
    "jarima",
    "nafaqa",
    "mulk",
    "vorislik",
    "ajrashish",
    "huquqiy",
    "qonuniy",
    "jinoiy",
    "notarius",
    "armiya",
    "armiga",
    "armiyaga",
    "harbiy",
    "soldat",
    "chaqiruv",
    "chaqiriq",
    "mudofaa",
    "askar",
    "oylik",
    "oylk",
    "zp",
    "maosh",
    "aliment",
    "ishdan",
    "haydad",
    "haydash",
    "dezertir",
    "povestka",
    "qamash",
    "qamoq",
    "urishib",
    "urish",
    "urib",
    "urdim",
    "kaltak",
    "janjal",
    "haqorat",
    "nalog",
    "shtraf",
    "militsiya",
    "polis",
    "politsiya",
    "hibs",
    "arest",
    "ogirlab",
    "ogirlash",
    "ogirlik",
    "ogrilik",
    "zalil",
)

# Leading boundary so "суд" does not fire inside "посуда", but stems still
# match inflections (зарплата, уволили, qamashadimi, armiga).
CYRILLIC_LEGAL_HINT = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(cue) for cue in CYRILLIC_LEGAL_CUES) + r")",
    re.IGNORECASE,
)
LATIN_LEGAL_HINT = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(cue) for cue in LATIN_LEGAL_CUES) + r")",
    re.IGNORECASE,
)

# "bormasam nma boladi" / "boʻlmasa nima boladi" is a legal-consequence
# question only when a legal stem is already in the same string. The
# pattern itself is extra recall for army/skip phrasing that names no code.
_CONSEQUENCE = re.compile(
    r"(?:bormasa\w*|bo[ʻʼ''`]?lmasa\w*|qamasha\w*)"
    r".{0,24}(?:nma|nima|nm)\s*boladi",
    re.IGNORECASE,
)


def has_cyrillic_legal_cue(text: str) -> bool:
    """True when `text` carries a Cyrillic legal stem from the tuple above."""
    return bool(text) and CYRILLIC_LEGAL_HINT.search(text) is not None


def has_latin_legal_cue(text: str) -> bool:
    """True when `text` carries a Latin / SMS legal stem."""
    if not text:
        return False
    return LATIN_LEGAL_HINT.search(text) is not None or bool(
        LATIN_LEGAL_HINT.search(norm(text))
    )


def has_legal_cue(text: str) -> bool:
    """Latin or Cyrillic legal vocabulary, including informal spellings."""
    if not text:
        return False
    if has_latin_legal_cue(text) or has_cyrillic_legal_cue(text):
        return True
    folded = norm(text)
    return bool(_CONSEQUENCE.search(text) or _CONSEQUENCE.search(folded))
