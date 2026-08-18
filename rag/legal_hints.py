"""Legal-vocabulary gate so statute questions are not answered as chat.

Latin stems used to live only as `_LEGAL_HINT` in pipeline.py. Informal
spellings (`armiga`, `qamashadimi`, `povestka`) never contained `modda` or
`armiya`, so the cosine fallback sent them to general chat — which then
invented jarima / qamash from memory. Cyrillic stems are the same gate for
Russian wages/firing stories.

Query normalization (armiga→harbiy xizmat, povestka→chaqiruv) still runs first in
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
    "бьёт",
    "бьет",
    "ударил",
    "украл",
    "краж",
    "развестись",
)
# "суд" is a cue, but "судно" (vessel) and "судак" are everyday words sharing
# the prefix -- it moves to the guarded list below.

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
    "konstitutsiya",
    "litsenziya",
    "jarima",
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
    "ishdan hayda",
    "ishdan boshat",
    "ishdan boʻshat",
    "ishdan chiqar",
    "haydad",
    "haydash",
    "ajrashish",
    "ajrashmoqchi",
    "ajrashaman",
    "qarz",
    "kredit",
    "advokat",
    "yurist",
    "uradi",
    "urdilar",
    "urdi",
    "urayapti",
    "tolov bermay",
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

# Cues whose bare prefix collides with everyday words: "davom etadi" and
# "davomat" are not davo (claim), "nafaqat" is not nafaqa (benefit), "sudoku"
# is not sud (court), "судно" (vessel) is not суд. The guard only blocks the
# specific colliding continuations; real inflections still match.
LATIN_GUARDED_CUES = (r"davo(?![ml])", r"nafaqa(?!t\b)", r"sud(?!o)")
CYRILLIC_GUARDED_CUES = (r"суд(?!н)",)

# Leading boundary so "суд" does not fire inside "посуда", but stems still
# match inflections (зарплата, уволили, qamashadimi, armiga).
CYRILLIC_LEGAL_HINT = re.compile(
    r"(?<!\w)(?:" + "|".join(
        [re.escape(cue) for cue in CYRILLIC_LEGAL_CUES] + list(CYRILLIC_GUARDED_CUES)
    ) + r")",
    re.IGNORECASE,
)
LATIN_LEGAL_HINT = re.compile(
    r"(?<!\w)(?:" + "|".join(
        [re.escape(cue) for cue in LATIN_LEGAL_CUES] + list(LATIN_GUARDED_CUES)
    ) + r")",
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
    return has_latin_legal_cue(text) or has_cyrillic_legal_cue(text)
