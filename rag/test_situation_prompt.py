"""Unit tests for the situation system prompt and Cyrillic legal cues.

No LLM, no GPU, no network.

    python test_situation_prompt.py
"""
from legal_hints import (
    CYRILLIC_LEGAL_CUES, CYRILLIC_LEGAL_HINT, has_cyrillic_legal_cue,
)
from situation_prompt import (
    BANNED_FINDINGS_UZ, DISCLAIMER_RU, DISCLAIMER_UZ,
    SITUATION_SYSTEM_RU, SITUATION_SYSTEM_UZ,
    audit_fail_reply, generation_system, mostly_cyrillic, situation_system_for,
)

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {got}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


print("1. UZ prompt carries the Day 2 contract")
check("disclaimer present", DISCLAIMER_UZ in SITUATION_SYSTEM_UZ, True)
for phrase in BANNED_FINDINGS_UZ:
    check(f"bans {phrase!r}", phrase in SITUATION_SYSTEM_UZ, True)
check("2–3 sentence facts", "2–3 jumlada" in SITUATION_SYSTEM_UZ, True)
check("candidates only from supplied text",
      "berilgan matndagi" in SITUATION_SYSTEM_UZ, True)
check("quote deadlines/duties/fines from the article",
      "muddatlar, majburiyatlar, jarimalar" in SITUATION_SYSTEM_UZ, True)
check("thin criminal facts: class of harm, not JK 110",
      "JK 110" in SITUATION_SYSTEM_UZ and "burun qonashi" in SITUATION_SYSTEM_UZ,
      True)


print("\n2. RU prompt is the Russian equivalent")
check("RU disclaimer", DISCLAIMER_RU in SITUATION_SYSTEM_RU, True)
check("RU bans закон нарушен", "закон нарушен" in SITUATION_SYSTEM_RU, True)
check("RU bans вы виновны", "вы виновны" in SITUATION_SYSTEM_RU, True)
check("RU bans ответственность finding",
      "вы будете нести ответственность" in SITUATION_SYSTEM_RU, True)
check("RU 2–3 sentences", "2–3 предложениях" in SITUATION_SYSTEM_RU, True)
check("RU thin facts not УК 110",
      "УК 110" in SITUATION_SYSTEM_RU and "кровь из носа" in SITUATION_SYSTEM_RU,
      True)


print("\n3. situation_system_for: UZ default, RU if mostly Cyrillic")
check("Latin Uzbek -> UZ",
      situation_system_for("Oyiga oylik bermadi, ishdan haydadi") is SITUATION_SYSTEM_UZ,
      True)
check("Russian wages -> RU",
      situation_system_for("Меня уволили и зарплату не выплатили") is SITUATION_SYSTEM_RU,
      True)
check("empty -> UZ", situation_system_for("") is SITUATION_SYSTEM_UZ, True)
check("digits only -> UZ", situation_system_for("123 456") is SITUATION_SYSTEM_UZ, True)
check("mostly_cyrillic Latin", mostly_cyrillic("oylik bermadi"), False)
check("mostly_cyrillic Russian", mostly_cyrillic("зарплату не дали"), True)
check("more Latin than Cyrillic stays UZ",
      mostly_cyrillic("hello world this is latin зарплата"), False)


print("\n4. Cyrillic legal cues (tuple + regex)")
check("cues are a tuple", isinstance(CYRILLIC_LEGAL_CUES, tuple), True)
check("regex is compiled", hasattr(CYRILLIC_LEGAL_HINT, "search"), True)
for q, want in [
    ("меня уволили с работы", True),
    ("зарплату за два месяца не дали", True),
    ("статья 253 трудового кодекса", True),
    ("что говорит закон", True),
    ("иш ҳақини бермади", True),
    ("если не пойду в армию", True),
    ("привет как дела", False),
    ("посуда на кухне", False),          # "суд" must not fire inside посуда
    ("oylik bermadi", False),            # Latin is pipeline._LEGAL_HINT's job
]:
    check(q, has_cyrillic_legal_cue(q), want)


print("\n5. prompt selection: long story vs named-article lookup")
from corpus_index import CorpusIndex
from situation_queries import issue_queries

LOOKUP = "LOOKUP-PROMPT-UNCHANGED"
idx = CorpusIndex([])
story = (
    "Ish beruvchi ikki oydan beri ish haqini bermayapti va meni "
    "ishdan haydab yubordi. Nima qilishim mumkin?"
)
jk = "JK 999-modda nima deydi?"
ru = "Работодатель два месяца не выплачивает зарплату и уволил меня. Что делать?"
check("long story has no named N-modda", bool(idx.parse_references(story)), False)
check("JK 999 has named refs", bool(idx.parse_references(jk)), True)
check("long story selects situation UZ",
      generation_system(idx.parse_references(story), LOOKUP, story)
      is SITUATION_SYSTEM_UZ, True)
check("JK 999 keeps lookup prompt",
      generation_system(idx.parse_references(jk), LOOKUP, jk), LOOKUP)
check("Russian labour selects RU prompt",
      generation_system([], LOOKUP, ru) is SITUATION_SYSTEM_RU, True)
check("Russian labour is a legal cue", has_cyrillic_legal_cue(ru), True)
qs = issue_queries(ru)
check("Russian labour search stays Latin-Uzbek Mehnat",
      any("Mehnat kodeksi" in q for q in qs), True)
check("Russian labour search has no Cyrillic query strings",
      all(not mostly_cyrillic(q) for q in qs), True)


print("\n6. audit-fail must not dump statute bodies")
fat = [{"code_title": "Mehnat kodeksi", "article_display": "253-modda",
        "text": "X" * 8000} for _ in range(6)]
uz_fail = audit_fail_reply(story, fat)
ru_fail = audit_fail_reply(ru, fat)
dump = "".join(a["text"] for a in fat)
check("UZ audit-fail is short", len(uz_fail) < 800, True)
check("RU audit-fail is short", len(ru_fail) < 800, True)
check("UZ audit-fail does not dump bodies", dump[:20] not in uz_fail, True)
check("RU audit-fail does not dump bodies", dump[:20] not in ru_fail, True)
check("RU audit-fail is Russian", "Я не юрист" in ru_fail, True)
check("UZ audit-fail has disclaimer", DISCLAIMER_UZ in uz_fail, True)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all situation_prompt tests pass")
