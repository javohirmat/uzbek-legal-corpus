"""Unit tests for lost-in-the-middle packing. No LLM, no GPU, no network.

    python test_pack_context.py
"""
from pack_context import format_grounding, pack_articles

fails = []


def check(label, got, want):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:56} -> {got}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


print("1. pack_articles: best first, second-best last")
six = list("ABCDEF")
check("6 articles", pack_articles(six), list("ACDEFB"))
check("2 articles unchanged", pack_articles(["A", "B"]), ["A", "B"])
check("1 article unchanged", pack_articles(["A"]), ["A"])
check("empty", pack_articles([]), [])
check("None -> empty", pack_articles(None), [])

original = [{"code": "a"}, {"code": "b"}, {"code": "c"}]
packed = pack_articles(original)
check("does not mutate input", [h["code"] for h in original], ["a", "b", "c"])
check("3 articles: best, rest, second", [h["code"] for h in packed], ["a", "c", "b"])


print("\n2. format_grounding: statutes then the user's story")
hits = [
    {"code_title": "Mehnat kodeksi", "article_display": "253-modda",
     "title": "Ish haqi", "text": "BEST article body"},
    {"code_title": "Mehnat kodeksi", "article_display": "161-modda",
     "text": "SECOND article body"},
    {"code": "fuqarolik_kodeksi_1qism", "article": "14",
     "text": "MIDDLE article body"},
]
story = "Oyiga oylik bermadi, keyin ishdan haydadi."
out = format_grounding(hits, story)
check("story after statutes", out.index("BEST article body") < out.index(story), True)
check("FOYDALANUVCHI VAZIYATI marks the story",
      "FOYDALANUVCHI VAZIYATI:\n" + story in out, True)
check("MODDALAR header first", out.startswith("MODDALAR:\n"), True)
check("best article is the first block",
      out.split("FOYDALANUVCHI")[0].find("BEST article body")
      < out.split("FOYDALANUVCHI")[0].find("SECOND article body"), True)
check("second-best is the last statute block",
      out.split("FOYDALANUVCHI")[0].rfind("SECOND article body")
      > out.split("FOYDALANUVCHI")[0].rfind("MIDDLE article body"), True)
check("corpus header shape", "[Mehnat kodeksi | 253-modda | Ish haqi]" in out, True)
check("thin retriever dict still formats",
      "[fuqarolik_kodeksi_1qism | 14]" in out, True)


print("\n3. schema flexibility and empty inputs")
thin = format_grounding(
    [{"code": "jinoyat_kodeksi", "article_id": "110", "body": "thin text"}],
    "burun qonadi",
)
check("article_id and body aliases",
      "[jinoyat_kodeksi | 110]" in thin and "thin text" in thin, True)
check("empty hits still restates the story",
      format_grounding([], "suv bosdi").endswith("suv bosdi"), True)
check("skips non-dicts", "modda" not in format_grounding(["not a hit"], "q"), True)


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all pack_context tests pass")
