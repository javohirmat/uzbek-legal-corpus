"""Unit tests for situational multi-query fusion. No LLM, no GPU, no network.

    python test_situation.py
"""
from situation_queries import cap_per_code, merge_queries, parse_query_list, queries_for, rrf_fuse

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


print("\n5. queries_for fallback when rewrite fails")

class _Expander:
    def expand(self, q):
        return q + " mehnatga haq tolash"

check("timeout/None complete -> original+expand only",
      queries_for("oylik bermadi", _Expander(), complete_fn=None),
      ["oylik bermadi mehnatga haq tolash"])
check("raising complete -> same fallback",
      queries_for("oylik bermadi", _Expander(),
                  complete_fn=lambda q: (_ for _ in ()).throw(TimeoutError())),
      ["oylik bermadi mehnatga haq tolash"])
check("good rewrite prepends original",
      queries_for("oylik bermadi", _Expander(),
                  complete_fn=lambda q: '["ish haqi muddati", "ishdan boshathish"]'),
      ["oylik bermadi mehnatga haq tolash", "ish haqi muddati", "ishdan boshathish"])


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all situation tests pass")
