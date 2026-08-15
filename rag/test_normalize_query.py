"""Unit tests for query-side spelling / SMS / dialect normalization.

No LLM, no GPU, no network.

    python test_normalize_query.py
"""
from normalize_query import (
    OKINA, looks_like_citation, normalize_query, transliterate_cyrillic,
)
from query_expand import QueryExpander
from situation_queries import issue_queries, queries_for

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {got!r}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


def contains(label, hay, needle):
    ok = needle in hay
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {hay!r}')
    if not ok:
        fails.append(f"{label}: {needle!r} not in {hay!r}")


print("1. apostrophe variants collapse to corpus okina")
want = "o" + OKINA + "zbek"
for q in (
    "o'zbek",
    "o‘zbek",
    "oʻzbek",
    "o\u2018zbek",
    "o\u2019zbek",
    "o\u02bbzbek",
):
    check(q, normalize_query(q), want)
check("g' vs gʻ", normalize_query("g'alaba")[:2], "g" + OKINA)


print("\n2. Cyrillic Uzbek → Latin-Uzbek")
check("ўзбек", normalize_query("ўзбек"), want)
check("transliterate ўзбек", transliterate_cyrillic("ўзбек"), want)
contains("ғ → gʻ", normalize_query("ғарб"), "g" + OKINA)
contains("Ғ keeps capital G", normalize_query("Ғарб"), "G" + OKINA)


print("\n3. SMS / chat shortenings (no citation in the string)")
got = normalize_query("oylk bermayapti")
contains("oylk → oylik", got, "oylik")
contains("oylk keeps bermayapti", got, "bermayapti")
check("nma → nima", normalize_query("nma boladi"), "nima boladi")
check("nm → nima", normalize_query("nm qilishadi"), "nima qilishadi")
contains("zp → oylik", normalize_query("zp bermadi"), "oylik")
contains("haydad → haydash", normalize_query("ishdan haydad"), "haydash")
contains("t6lov → toʻlov", normalize_query("t6lov"), "to" + OKINA + "lov")
contains("6zbek → oʻzbek", normalize_query("6zbek"), want)
contains("wunaqa → shunaqa", normalize_query("wunaqa"), "shunaqa")
contains("nimaaa → nima", normalize_query("nimaaa boladi"), "nima")
contains("nimaboladi spaced", normalize_query("nimaboladi"), "nima boladi")
contains("armiga → armiya", normalize_query("armiga bormasam"), "armiya")
contains("ogirlab → o‘g‘irlik", normalize_query("ogirlab ketishdi"), "g" + OKINA + "irlik")


print("\n4. citations: digit SMS must not smash 999 / 253 / 141²")
jk = normalize_query("JK 999-modda nima deydi?")
contains("JK survives", jk, "JK")
contains("999-modda survives", jk.lower(), "999-modda")
check("looks like citation", looks_like_citation("JK 999-modda nima deydi?"), True)
check("253-modda unchanged", "253-modda" in normalize_query("Mehnat 253-modda"), True)
sup = normalize_query("141²-modda")
contains("superscript ² survives", sup, "141²")
check("4801-modda keeps 4", "4801-modda" in normalize_query("4801-modda"), True)
check("2 oy is not SMS 6/4", "2 oy" in normalize_query("oylk 2 oy bermayapti"), True)
check("non-citation allows 6-map", looks_like_citation("oylk 2 oy bermayapti"), False)


print("\n5. Russian chat → Latin search tokens")
ru_army = normalize_query("Если не пойду в армию что будет?")
contains("армию → armiya", ru_army.lower(), "armiya")
check("no leftover Cyrillic in army query",
      any("а" <= ch.lower() <= "я" or ch in "ўқғҳ" for ch in ru_army), False)
ru_pay = normalize_query("зп не дали и уволили")
contains("зп → oylik", ru_pay, "oylik")
contains("уволили → ishdan", ru_pay, "ishdan")
contains("алименты → aliment", normalize_query("алименты не платит"), "aliment")
contains("срочка → muddatli", normalize_query("срочка"), "muddatli")


print("\n6. expander + situation templates see the normalized form")
exp = QueryExpander.load("synonyms.json")
lab = "oylk 2 oy bermayapti ishdan haydad nima boladi"
expanded = exp.expand(lab)
contains("labour expand mentions ish haqi", expanded, "ish haqini toʻlash")
contains("labour expand mentions bekor qilish", expanded, "bekor qilish")
issues = issue_queries(lab)
check("wages issue after oylk",
      any("ish haqini toʻlash" in q for q in issues), True)
check("dismissal issue after haydad",
      any("bekor qilish" in q for q in issues), True)

army = exp.expand("Armiyaga bormadim nima qilishadi")
contains("army expand has chaqiruv", army, "chaqiruv")
contains("army expand has harbiy xizmat", army, "harbiy xizmat")
army_issues = issue_queries("Harbiy xizmatga bormasam nima bo‘ladi?")
check("military issue template",
      any("harbiy xizmat" in q.lower() or "chaqiruv" in q for q in army_issues), True)

qs_ru = queries_for(
    "Если не пойду в армию что будет?",
    exp,
    complete_fn=None,
)
check("Russian army does not search raw Cyrillic",
      any("арми" in q or "пойду" in q for q in qs_ru), False)
check("Russian army searches Latin military phrasing",
      any("harbiy" in q or "chaqiruv" in q or "armiya" in q.lower() for q in qs_ru),
      True)


print("\n7. idempotent on already-clean Latin")
clean = "Harbiy xizmatga bormasam nima boʻladi?"
check("clean military is stable", normalize_query(clean), normalize_query(normalize_query(clean)))
check("empty", normalize_query(""), "")


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all normalize_query tests pass")
