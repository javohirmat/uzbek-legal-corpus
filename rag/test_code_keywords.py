"""Unit tests for per-code colloquial keyword boost. No LLM, no GPU, no network.

    python test_code_keywords.py
"""
from code_keywords import (
    CodeKeywords, boost_matched_codes, cap_for_matches, keyword_queries,
    match_code_slugs, pretty_code,
)
from query_expand import QueryExpander
from situation_queries import queries_for

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {got!r}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


cat = CodeKeywords.load()
exp = QueryExpander.load("synonyms.json")

print(f"catalog: {len(cat)} codes\n")

print("1. every local slug has a keyword object")
import glob, json, os
import config as C
slugs = set()
for p in glob.glob(os.path.join(C.LOCAL_ARTICLES, "*.jsonl")):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                slugs.add(json.loads(line)["code"])
                break
check("25 local codes", len(slugs), 25)
check("catalog covers every slug", set(e["code"] for e in cat.entries), slugs)
for e in cat.entries:
    check(f"{e['code']} has match patterns", len(e["pats"]) >= 3, True)
    check(f"{e['code']} has formal phrases", len(e["formal"]) >= 3, True)


print("\n2. army slang → jinoyat / ma'muriy (not a random code)")
army = match_code_slugs("armiyaga bormasa Nima boladi")
check("army extra uses 225/237 title, not hisobdan o'tish",
      any("muqobil xizmatdan bo" in q.lower() or "muqobil xizmatdan bo" in q
          for q in keyword_queries("armiyaga bormasa Nima boladi")), True)
check("army extra does not prefer MJK 235 title",
      any("hisobdan" in q for q in keyword_queries("armiyaga bormasa Nima boladi")),
      False)
check("armiya hits jinoyat", "jinoyat_kodeksi" in army, True)
check("armiya hits mamuriy", "mamuriy_javobgarlik_kodeksi" in army, True)
check("armiya does not hit havo", "havo_kodeksi" in army, False)
check("armiya does not hit soliq", "soliq_kodeksi" in army, False)
armiga = match_code_slugs("aka povestka keldi armiga bormasam nma boladi")
check("armiga SMS still hits jinoyat", "jinoyat_kodeksi" in armiga, True)


print("\n3. havo / asosiy qonun route; futbol does not")
check("havo → havo_kodeksi",
      "havo_kodeksi" in match_code_slugs("havo nima"), True)
check("asosiy qonun → konstitutsiya",
      "konstitutsiya" in match_code_slugs("asosiy qonun nima deydi"), True)
check("konstitutsiya → konstitutsiya",
      "konstitutsiya" in match_code_slugs("konstitutsiya"), True)
check("futbol nima matches no code", match_code_slugs("futbol nima"), [])
check("salom aka matches no code", match_code_slugs("salom aka qalaysan"), [])


print("\n4. theft stays Oʻgʻrilik; wages stay Mehnat, not Soliq 371")
phone_qs = keyword_queries("telefonimni ogirlab ketishdi nma qilaman aka")
blob = " ".join(phone_qs).lower()
check("phone theft extra names jinoyat",
      any("jinoyat" in q.lower() for q in phone_qs), True)
check("phone theft extra has o‘g‘rilik / yashirin",
      "gʻrilik" in blob or "grilik" in blob or "yashirin" in blob, True)
check("phone theft extra does not name talonchilik",
      "talonchilik" not in blob, True)

wage_slugs = match_code_slugs("oylk 2 oydan beri yoq zp bermayapti")
check("zp/oylk hits mehnat", "mehnat_kodeksi" in wage_slugs, True)
check("zp/oylk does not hit soliq", "soliq_kodeksi" in wage_slugs, False)
wage_extra = " ".join(keyword_queries("zp bermayapti")).lower()
check("wage extra has ish haqi timing", "ish haqi" in wage_extra or "toʻlash" in wage_extra
      or "tolash" in wage_extra, True)
check("wage extra does not inject mehnatga haq toʻlash",
      "mehnatga haq" not in wage_extra, True)
check("nalog hits soliq",
      "soliq_kodeksi" in match_code_slugs("nalog tolamadim nma boladi"), True)


print("\n5. boost is extra queries, never the only path; JK 999 still parses")
qs = queries_for("armiyaga bormasa Nima boladi", exp, complete_fn=None)
check("army still has original/expand first", bool(qs) and bool(qs[0]), True)
check("army extra did not drop the user query path",
      any("armiya" in q.lower() or "harbiy" in q.lower() or "chaqiruv" in q
          for q in qs), True)
unknown = queries_for("noma'lum savol", exp, complete_fn=None)
check("unmatched question still searches original", len(unknown) >= 1, True)
check("futbol does not force a code-family query",
      not any("kodeksi" in q.lower() or "kodeks" in q.lower()
              for q in queries_for("futbol nima", None, complete_fn=None)[1:]),
      True)

from corpus_index import CorpusIndex as CI
records = []
for p in sorted(glob.glob(os.path.join(C.LOCAL_ARTICLES, "*.jsonl"))):
    with open(p, encoding="utf-8") as f:
        records += [json.loads(l) for l in f if l.strip()]
ci = CI(records)
st, hint, _ = ci.resolve(ci.parse_references("JK 999-modda nima deydi?")[0])
check("JK 999 still OUT_OF_RANGE", st, "OUT_OF_RANGE")
check("JK 999 hint names jinoyat", "jinoyat_kodeksi" in hint.get("slugs", []), True)


print("\n6. RRF boost + cap: one code may fill 6; several keep cap 2")
check("two army codes keep per-code cap",
      cap_for_matches(["jinoyat_kodeksi", "mamuriy_javobgarlik_kodeksi"]), 2)
check("havo alone may occupy the window",
      cap_for_matches(["havo_kodeksi"]), 6)
check("no matches keep default cap", cap_for_matches([]), 2)

keys = [
    ("suv_kodeksi", "72"), ("jinoyat_kodeksi", "225"),
    ("mamuriy_javobgarlik_kodeksi", "237"), ("suv_kodeksi", "73"),
]
boosted = boost_matched_codes(
    keys, ["jinoyat_kodeksi", "mamuriy_javobgarlik_kodeksi"])
check("boosted codes rise without inventing keys",
      boosted[:2],
      [("jinoyat_kodeksi", "225"), ("mamuriy_javobgarlik_kodeksi", "237")])
check("boost does not drop neighbours",
      ("suv_kodeksi", "72") in boosted, True)
check("pretty slug", pretty_code("fuqarolik_kodeksi_1qism"), "Fuqarolik kodeksi")


print("\n7. queries_for still refuses Soliq 371 / Talonchilik phrasing")
qs_wage = queries_for("зп не платят уволили", exp, complete_fn=None)
check("Russian zp searches Mehnat",
      any("Mehnat" in q or "ish haqi" in q for q in qs_wage), True)
check("Russian zp does not search Soliq",
      any("Soliq" in q or "soliq" in q for q in qs_wage), False)
qs_phone = queries_for("telefonimni ogirlab ketishdi", exp, complete_fn=None)
check("phone queries_for has no Talonchilik",
      any("talonchilik" in q.lower() for q in qs_phone), False)
check("synonyms.json untouched by this module",
      "talonchilik" not in open("synonyms.json", encoding="utf-8").read().lower(),
      True)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all code_keywords tests pass")
