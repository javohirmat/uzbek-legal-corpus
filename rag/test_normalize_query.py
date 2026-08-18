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
contains("armiga → harbiy xizmat (armiya: 0 corpus postings)",
         normalize_query("armiga bormasam"), "harbiy xizmat")


print("\n3b. counts glued to digits are never letter-play (4ta, 6oy)")
check("4ta survives", normalize_query("4ta bola bor aliment"), "4ta bola bor aliment")
check("6ta survives (oʻta is a different real word)",
      normalize_query("6ta shartnoma"), "6ta shartnoma")
check("6oy survives", "6oy" in normalize_query("2 yil 6oy"), True)
check("4chi survives", "4chi" in normalize_query("4chi marta"), True)
check("4kishi survives", "4kishi" in normalize_query("4kishi ishdan boʻshatildi"), True)
check("A4 product code survives", "A4" in normalize_query("avtomobil A4 rusumli"), True)
check("4x4 survives", "4x4" in normalize_query("kompaniya 4x4"), True)


print("\n3c. English tokens in Uzbek chat survive the w-map")
check("whatsapp survives", "whatsapp" in normalize_query("whatsapp orqali tahdid qilishdi"), True)
check("web survives", "web" in normalize_query("web sayt"), True)
check("windows survives", "windows" in normalize_query("windows"), True)
check("power bank survives", "power bank" in normalize_query("power bank kerak"), True)
check("workflow survives", "workflow" in normalize_query("workflow"), True)
check("wunaqa still maps", normalize_query("wunaqa holatda"), "shunaqa holatda")
check("Wu still maps", normalize_query("Wu nimadir"), "Shu nimadir")


print("\n3d. www survives the w-map and triple-collapse, and stays idempotent")
check("www stable", normalize_query("www.lex.uz"), "www.lex.uz")
check("www idempotent",
      normalize_query(normalize_query("www.lex.uz")), normalize_query("www.lex.uz"))
contains("ogirlab → o‘g‘rilik", normalize_query("ogirlab ketishdi"), "g" + OKINA + "rilik")


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
contains("армию → harbiy xizmat (statute token)", ru_army.lower(), "harbiy xizmat")
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
check("labour expand does not inject Soliq 371 phrasing",
      "mehnatga haq toʻlash" not in expanded, True)

phone = exp.expand("telefonimni ogirlab ketishdi nma qilaman aka")
contains("theft expand uses statute Oʻgʻrilik", phone, "g" + OKINA + "rilik")
contains("theft expand uses yashirin ravishda", phone, "yashirin")
check("theft expand does not inject Talonchilik (JK 166)",
      "talonchilik" not in phone.lower(), True)

zp_qs = queries_for("зп не платят уволили", exp, complete_fn=None)
check("зп slang searches Mehnat pay-timing",
      any("ish haqini toʻlash" in q or "Mehnat" in q for q in zp_qs), True)
check("зп slang does not search Soliq 371 phrasing",
      any("mehnatga haq toʻlash" in q or "Soliq" in q for q in zp_qs), False)
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


print("\n8. Telegram latin: w=sh, c=ch (reform letters are just-in-case)")
# w/c are the spellings customers actually send. A wu-only map leaves
# iwxona / ketiwdi / qamawadimi looking like they never mentioned ish / shartnoma.
contains("iwxonada → ishxonada", normalize_query("iwxonada oylk bermayapti"), "ishxonada")
contains("iwxonada still maps oylk", normalize_query("iwxonada oylk bermayapti"), "oylik")
contains("ketiwdi → ketishdi", normalize_query("ogirlab ketiwdi"), "ketishdi")
contains("kocada → kochada", normalize_query("kocada urib yubordim"), "kochada")
contains("qanca → qancha", normalize_query("qanca undiriladi"), "qancha")
contains("haydamoqci → haydamoqchi", normalize_query("iwdan haydamoqci"), "haydamoqchi")
contains("iwdan → ishdan", normalize_query("iwdan haydamoqci"), "ishdan")
contains("qamawadimi → qamashadimi", normalize_query("kocada urib yubordim meni qamawadimi"), "qamashadimi")
contains("wartnoma → shartnoma", normalize_query("wartnoma buzildi"), "shartnoma")
contains("bowqa → boshqa", normalize_query("bowqa model beraman"), "boshqa")
contains("ş → sh (rare reform letter, still maps)", normalize_query("işxonada oylik yoq"), "ishxonada")
contains("ç → ch", normalize_query("çiqarish tartibi"), "chiqarish")
contains("ō → oʻ", normalize_query("ōzbek"), want)
contains("ğ → gʻ", normalize_query("ğalaba")[:2], "g" + OKINA)
contains("ñ → ng", normalize_query("soñgi oylik"), "songgi")
contains("muzlatgic oldim → muzlatgich", normalize_query("muzlatgic oldim"), "muzlatgich")
contains("sotuvci endi → sotuvchi", normalize_query("sotuvci endi"), "sotuvchi")
contains("hec qanday → hech", normalize_query("hec qanday ogohlantirishsiz"), "hech")
check("telegram wage is idempotent",
      normalize_query("iwxonada oylk"), normalize_query(normalize_query("iwxonada oylk")))

print("\n8b. English / URL tokens must survive the broader w/c maps")
check("whatsapp still survives", "whatsapp" in normalize_query("whatsapp orqali tahdid qilishdi"), True)
check("facebook survives c-map", "facebook" in normalize_query("facebook orqali haqorat"), True)
check("camera survives c-map", "camera" in normalize_query("camera jarima keldi"), True)
check("card survives c-map", "card" in normalize_query("bank card yeb qolishdi"), True)
check("kiwi is not kishi", "kiwi" in normalize_query("kiwi yedim"), True)
check("new stays new", "new" in normalize_query("new telefon oldim"), True)
check("www.lex.uz still stable", normalize_query("www.lex.uz"), "www.lex.uz")
check("vk.com survives", "vk.com" in normalize_query("vk.com da tahdid"), True)
jk_w = normalize_query("JK 169-modda ketiwdi nima jazo")
contains("citation digits survive next to iw", jk_w, "169-modda")
contains("ketiwdi still maps beside JK", jk_w, "ketishdi")

print("\n8c. expander sees Telegram spelling as formal labour / theft / street")
tg_wage = "iwxonada 2 oydan beri oylk bermayapti nma qilaman"
expanded_tg = exp.expand(tg_wage)
contains("telegram wage expand mentions ish haqi", expanded_tg, "ish haqini toʻlash")
contains("telegram wage expand mentions ishxonada", expanded_tg, "ishxonada")
tg_qs = queries_for(tg_wage, exp, complete_fn=None)
check("telegram wage searches Mehnat pay-timing",
      any("ish haqini toʻlash" in q or "Mehnat" in q for q in tg_qs), True)
phone_w = exp.expand("telefonimni ogirlab ketiwdi nma qilaman aka")
contains("telegram theft still Oʻgʻrilik", phone_w, "g" + OKINA + "rilik")
issues_c = issue_queries("kocada urib yubordim meni qamawadimi")
check("telegram street still has beating issue",
      any("urish" in q or "haqorat" in q or "tana shikast" in q or "Maʼmuriy" in q
          for q in issues_c), True)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all normalize_query tests pass")
