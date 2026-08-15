"""Unit tests for situational multi-query fusion. No LLM, no GPU, no network.

    python test_situation.py
"""
from situation_queries import (
    cap_per_code, issue_queries, merge_queries, pack_lost_in_middle,
    parse_query_list, queries_for, rrf_fuse,
)

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {got}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


print("1. RRF fuse across query lists")
a = [("suv_kodeksi", "72"), ("suv_kodeksi", "73"), ("fuqarolik_kodeksi_1qism", "14")]
b = [("fuqarolik_kodeksi_1qism", "14"), ("fuqarolik_kodeksi_2qism", "599"), ("suv_kodeksi", "72")]
fused = rrf_fuse([a, b])
check("FK 14 ranks above Suv 73 (in both lists vs one)",
      fused.index(("fuqarolik_kodeksi_1qism", "14"))
      < fused.index(("suv_kodeksi", "73")), True)
check("Suv 72 still present (also in both lists)",
      ("suv_kodeksi", "72") in fused, True)
check("empty lists fuse to empty", rrf_fuse([]), [])
check("single list preserves order", rrf_fuse([a]), a)


print("\n2. per-code cap in the final 6")
flooded = [
    ("suv_kodeksi", str(n)) for n in range(70, 80)
] + [
    ("fuqarolik_kodeksi_1qism", "14"),
    ("fuqarolik_kodeksi_2qism", "599"),
    ("mehnat_kodeksi", "253"),
]
capped = cap_per_code(flooded, cap=2, limit=6)
codes = [k[0] for k in capped]
check("at most 2 Suv", codes.count("suv_kodeksi"), 2)
check("FK 14 survives the Suv pile-up",
      ("fuqarolik_kodeksi_1qism", "14") in capped, True)
check("window size 6 or less", len(capped) <= 6, True)
check("cap 2 on a 3-code mix fills from later codes",
      cap_per_code(
          [("suv_kodeksi", "1"), ("suv_kodeksi", "2"), ("suv_kodeksi", "3"),
           ("fuqarolik_kodeksi_1qism", "14"), ("mehnat_kodeksi", "253")],
          cap=2, limit=6,
      ),
      [("suv_kodeksi", "1"), ("suv_kodeksi", "2"),
       ("fuqarolik_kodeksi_1qism", "14"), ("mehnat_kodeksi", "253")])


print("\n3. query merge: original first, no invented article numbers")
merged = merge_queries(
    "suv bosdi ijara",
    ["ijara shartnomasi zarar qoplash",
     "Suv kodeksi 72-modda",
     "suv bosdi ijara",
     "mulkka yetkazilgan zarar"],
    limit=5,
)
check("original is query #1", merged[0], "suv bosdi ijara")
check("invented 72-modda dropped", any("72" in q for q in merged), False)
check("duplicate of original dropped", merged.count("suv bosdi ijara"), 1)
check("kept civil-law rewrite", "ijara shartnomasi zarar qoplash" in merged, True)


print("\n4. parse rewrite JSON, including fences and garbage")
check("plain array",
      parse_query_list('["ijara zarar", "suv toshqini"]'),
      ["ijara zarar", "suv toshqini"])
check("fenced json",
      parse_query_list('```json\n["a", "b"]\n```'),
      ["a", "b"])
check("object with queries key",
      parse_query_list('{"queries": ["x", "y"]}'),
      ["x", "y"])
check("garbage -> empty", parse_query_list("not json at all"), [])
check("empty -> empty", parse_query_list(""), [])


print("\n5. queries_for: templates first, 27B only if still thin")

class _Expander:
    def expand(self, q):
        return q + " mehnatga haq tolash"

called = []
qs = queries_for(
    "Uyimni ijaraga olganman, kecha suv bosdi",
    _Expander(),
    complete_fn=lambda q: called.append(q) or '["should not run"]',
)
check("flood+lease skips 27B", called, [])
check("flood+lease keeps original first", qs[0].startswith("Uyimni ijaraga"), True)
check("flood+lease has ijara issue",
      any("ijara shartnomasi" in q for q in qs), True)
check("flood+lease has zarar issue",
      any("zarar qoplash" in q for q in qs), True)

check("wages template without 27B",
      "ish haqini toʻlash muddatlari Mehnat kodeksi"
      in queries_for("oylik bermadi", _Expander(), complete_fn=None), True)
_NOMA = "noma\u02bclum savol mehnatga haq tolash"
check("raising complete still keeps original+expand",
      queries_for("noma'lum savol", _Expander(),
                  complete_fn=lambda q: (_ for _ in ()).throw(TimeoutError()))[0],
      _NOMA)
check("thin rewrite still prepends original",
      queries_for("noma'lum savol", _Expander(),
                  complete_fn=lambda q: '["fuqarolik zarar", "mehnat haq"]')[0],
      _NOMA)
check("SMS oylk+haydad still emits Mehnat issues",
      issue_queries("oylk bermayapti ishdan haydad"),
      ["ish haqini toʻlash muddatlari Mehnat kodeksi",
       "mehnat shartnomasini bekor qilish Mehnat kodeksi"])
check("army slang emits JK chaqiruv issue",
      any("chaqiruv" in q for q in issue_queries("Armiyaga bormadim nima qilishadi")),
      True)
check("armiga SMS still emits chaqiruv issue",
      any("chaqiruv" in q for q in issue_queries(
          "aka povestka keldi armiga bormasam nma boladi")),
      True)
check("school fight emits shikast/Maʼmuriy issue",
      any("shikast" in q or "Maʼmuriy" in q
          for q in issue_queries("maktabda urishib qoldik meni qamashadimi")),
      True)
check("issue_queries splits lease vs flood",
      issue_queries("ijaraga olganman, suv bosdi"),
      ["ijara shartnomasi Fuqarolik kodeksi",
       "mulkka yetkazilgan zarar qoplash Fuqarolik kodeksi"])
check("Russian wages+firing still emits Latin Mehnat queries",
      issue_queries("два месяца не платит зарплату и уволил"),
      ["ish haqini toʻlash muddatlari Mehnat kodeksi",
       "mehnat shartnomasini bekor qilish Mehnat kodeksi"])
qs_ru = queries_for(
    "Работодатель два месяца не выплачивает зарплату и уволил меня",
    _Expander(),
    complete_fn=None,
)
check("Russian story does not search the raw Cyrillic",
      any("Работодатель" in q or "зарплат" in q for q in qs_ru), False)
check("Russian story still searches Latin Mehnat",
      any("Mehnat kodeksi" in q for q in qs_ru), True)


print("\n6. lost-in-the-middle packing: best first, second-best last")
six = list("ABCDEF")
check("6 articles", pack_lost_in_middle(six), list("ACDEFB"))
check("2 articles unchanged", pack_lost_in_middle(["A", "B"]), ["A", "B"])
check("1 article unchanged", pack_lost_in_middle(["A"]), ["A"])
check("empty", pack_lost_in_middle([]), [])


print("\n7. situation SYSTEM prompt contract")
src = open("pipeline.py", encoding="utf-8").read()
try:
    src += open("situation_prompt.py", encoding="utf-8").read()
except FileNotFoundError:
    pass
for needle in (
    "Aka",
    "Baraka toping",
    "davlat axborot",
    "rasmiy yozma",
    "Men yurist emasman",
    "qonun buzilgan",
    "qonun buzilishi",
    "siz aybdorsiz",
    "javobgar boʻlasiz",
    "oqibat",
):
    check(f"prompt mentions {needle!r}", needle in src, True)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all situation tests pass")
