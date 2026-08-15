"""Cyrillic legal cues the Latin `_LEGAL_HINT` in pipeline.py does not cover.

A Russian (or Uzbek-Cyrillic) wages/firing story never contains `modda` or
`ish haqi`, so the current keyword gate routes it to general chat. These
stems are extra signal for that gate — import later; do not edit pipeline
from this module.

Exported both as a tuple (easy to extend) and as a compiled regex.
"""
import re

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
)

# Leading boundary so "суд" does not fire inside "посуда", but stems still
# match inflections (зарплата, уволили).
CYRILLIC_LEGAL_HINT = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(cue) for cue in CYRILLIC_LEGAL_CUES) + r")",
    re.IGNORECASE,
)


def has_cyrillic_legal_cue(text: str) -> bool:
    """True when `text` carries a Cyrillic legal stem from the tuple above."""
    return bool(text) and CYRILLIC_LEGAL_HINT.search(text) is not None
