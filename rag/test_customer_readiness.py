"""Regressions for the bugs found in the pre-customer audit.

No LLM, no GPU, no network -- every path here is deterministic. Each section
names the failure it locks down, in the words of the customer question that
triggered it.

  1. quantities after a code name are not article lookups (every Uzbek case
     suffix, not just the ablative that used to be listed)
  2. a nonexistent insert article refuses instead of serving the base article
  3. Russian citation order and Russian code names resolve
  4. the shipped overrides.json cannot answer a statute question
  5. `not` guards survive Uzbek inflection
  6. a broken TOMARIS_API_KEYS fails closed instead of opening the box
  7. limits are parsed strictly; keys may contain ':'; keys meter per key
  8. a hand-edited usage file cannot 500 every request
  9. k -> g softening ("oyligimni") keeps the wage path alive
 10. tax questions keep the Tax Code; assault/divorce reach a template

    python test_customer_readiness.py
"""
import glob
import json
import os
import tempfile
import threading

from answer_rules import Overrides
from api_keys import KeyAuth, KeyAuthError
from code_keywords import match_code_slugs
from corpus_index import CorpusIndex
from legal_hints import has_legal_cue
from normalize_query import normalize_query
from query_expand import QueryExpander
from situation_queries import (
    exclude_unmentioned_soliq, issue_queries, queries_for,
)

fails = []


def check(label, got, want=True):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {label:58} -> {got!r}')
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


HERE = os.path.dirname(os.path.abspath(__file__))
_rows = [
    json.loads(line)
    for path in sorted(glob.glob(os.path.join(HERE, "..", "data", "articles", "*.jsonl")))
    for line in open(path, encoding="utf-8")
    if line.strip()
]
idx = CorpusIndex(_rows)
expander = QueryExpander.load(os.path.join(HERE, "synonyms.json"))


def refs_for(question):
    """Mirror LegalRAG._refs: raw first, then the normalized form.

    Cyrillic only becomes matchable after transliteration, so a test that calls
    parse_references() directly does not exercise the path the server takes.
    """
    refs = idx.parse_references(question)
    if refs:
        return refs
    normalized = normalize_query(question)
    return idx.parse_references(normalized) if normalized != question else refs


def served(question):
    """(status, article_display) for the first parsed reference, or (None, None)."""
    refs = refs_for(question)
    if not refs:
        return (None, None)
    status, _, record = idx.resolve(refs[0])
    return (status, record["article_display"] if record else None)


def state_path():
    return os.path.join(tempfile.mkdtemp(prefix="readiness-"), "usage.json")


def key_auth(raw, path=None):
    return KeyAuth.from_env({"TOMARIS_API_KEYS": raw,
                             "API_USAGE_FILE": path or state_path()})


class Req:
    def __init__(self, token=None):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}


print("1. a quantity after a code name is never an article")
# Only "3 oydan" used to be listed, so every other Uzbek case suffix produced a
# phantom lookup -- and a non-empty ref list SUPPRESSES situation retrieval, so
# the wage articles were never even searched.
for question in [
    "Mehnat kodeksi bo'yicha 3 oyga oylik bermayapti",
    "Mehnat kodeksi bo'yicha 3 oydan beri oylik bermayapti",
    "Mehnat kodeksi 15 kunga ta'til",
    "Mehnat kodeksi 15 kungacha",
    "Mehnat kodeksi 8 soatlik ish kuni",
    "Jinoyat kodeksi 14 yoshdan javobgarlik boshlanadimi",
    "Jinoyat kodeksi 18 yoshgacha",
    "Mehnat kodeksi 5 nafar xodim",
    "Mehnat kodeksi 2 million so'm",
    "Oila kodeksi 10 yilga yaqin",
    "Jinoyat kodeksi bo'yicha 100 baravar jarima",
    "Mehnat kodeksi bo'yicha 3 месяца зарплату не платят",
]:
    check(f"story, not a lookup: {question[:44]}", refs_for(question), [])

print("\n   real citations still resolve")
for question, want in [
    ("JK 169-modda", "169-modda"),
    ("JK 169", "169-modda"),
    ("Mehnat kodeksi 253-modda", "253-modda"),
    ("MK 253", "253-modda"),
    ("fuqarolik kodeksi 674 kerak", "674-modda"),
    ("JK 141.2-modda", "141²-modda"),
    ("JK 141²-modda", "141²-modda"),
    ("Saylov kodeksi 8¹-modda", "8¹-modda"),
    # the guard used to be anchored past the separating space, so the digits of
    # the NEXT article number rejected the first one
    ("JK 169 170 farqi nima", "169-modda"),
]:
    check(f"resolves: {question}", served(question)[1], want)


print("\n2. a nonexistent insert refuses instead of serving the base article")
# resolve()'s flat-digit fallback exists for "4801-modda" -> 480¹. It also fired
# when the user explicitly asked for an insert, answering "JK 169⁵" with JK 169
# (Oʻgʻrilik) as though it were the article requested.
for question in ["JK 169.5-modda", "JK 169⁵-modda", "MJK 27.2-modda"]:
    status, display = served(question)
    check(f"no base-article substitution: {question}", display, None)
    check(f"  and it is reported missing: {question}",
          status in ("REPEALED", "OUT_OF_RANGE"), True)
check("existing insert still resolves", served("JK 141.2-modda")[1], "141²-modda")
check("out-of-range still refuses", served("JK 999-modda")[0], "OUT_OF_RANGE")


print("\n3. Russian citations resolve (the partner's audience writes Russian)")
# "статья N УК" is the standard Russian order; the bare-number scan only looked
# FORWARD from the code name, so the deterministic path was lost entirely.
for question, want_code in [
    ("УК 169", "jinoyat_kodeksi"),
    ("ст. 169 УК", "jinoyat_kodeksi"),
    ("статья 169 УК", "jinoyat_kodeksi"),
    ("ст 253 ТК", "mehnat_kodeksi"),
    ("Уголовный кодекс статья 169", "jinoyat_kodeksi"),
    ("Трудовой кодекс 253 статья", "mehnat_kodeksi"),
    ("УПК 100", "jinoyat_protsessual_kodeksi"),
    ("КоАО 49", "mamuriy_javobgarlik_kodeksi"),
]:
    refs = refs_for(question)
    got = refs[0]["slugs"] if refs else None
    check(f"{question:30} -> {want_code}", bool(got) and want_code in got, True)
# A Russian quantity must still not become an article.
check("Russian quantity stays a story",
      refs_for("Трудовой кодекс 3 месяца не платят"), [])


print("\n4. the shipped overrides.json answers no statute question")
# It shipped two live demo bank rules, checked before parsing, retrieval and the
# model. "Ish vaqti" is the title of a whole Mehnat kodeksi chapter.
shipped = Overrides.load(os.path.join(HERE, "overrides.json"))
check("overrides.json ships empty", len(shipped.rules), 0)
for question in [
    "Ish vaqti haftasiga necha soat boʻlishi kerak?",
    "Tungi ish vaqti qanday hisoblanadi",
    "kredit foizi qancha",
    "ish tartibi qanday belgilanadi",
]:
    check(f"no canned answer: {question[:42]}", shipped.match(question), None)
for rule in shipped.rules:                      # holds if a demo is ever re-added
    labelled = "namuna" in (rule["answer"] + " " + rule["source"]).lower()
    check(f"rule '{rule['id']}' labelled as sample", labelled, True)


print("\n5. `not` guards survive Uzbek inflection")
# `not` was whole-word only, so "kodeks" never matched "kodeksi" -- the form
# every Uzbek speaker actually types.
guarded = Overrides([{
    "id": "demo", "any": ["kredit foizi"],
    "not": ["ipoteka", "kodeks", "qonun", "modda", "sud"],
    "answer": "namuna", "source": "namuna",
}])
check("plain trigger still fires", bool(guarded.match("kredit foizi qancha")), True)
for question in [
    "Fuqarolik kodeksi bo'yicha kredit foizi",
    "kredit foizi qonunda belgilanganmi",
    "kredit foizi kodeksda yozilganmi",
    "kredit foizi haqida qaysi moddada yozilgan",
    "kredit foizi bo'yicha sudga bersam",
    "ipoteka kredit foizi qancha",
]:
    check(f"blocked: {question[:44]}", guarded.match(question), None)


print("\n6. a broken TOMARIS_API_KEYS fails closed")
# Zero parsed keys used to mean open mode: the box answered everyone, unmetered,
# while the operator believed he had just handed out a capped trial key.
for raw in ["azizbek", "azizbek:", ":secret:10", "azizbek:t7-A:200/day",
            "azizbek:t7-A:1k", "azizbek:t7-A:-5"]:
    try:
        key_auth(raw)
        check(f"refuses to start open: {raw!r}", "started open", "SystemExit")
    except SystemExit:
        check(f"refuses to start open: {raw!r}", True)
try:
    key_auth("a:t7-A:200;b:t7-B:50")
    check("';' separator refused", "accepted", "SystemExit")
except SystemExit:
    check("';' separator refused", True)
check("empty setting is still open mode", key_auth("").enabled, False)
check("empty setting: authorize is a no-op", key_auth("").authorize(Req()), None)


print("\n7. limits parse strictly; keys may contain ':'; metering is per key")
auth = key_auth('"azizbek:t7-A:200"')           # env files keep the quotes
check("surrounding quotes stripped", auth.limit_for("azizbek"), 200)

auth = key_auth("azizbek:sk-tomaris:live:abc123:200")
check("colon-bearing key registered whole",
      auth.authorize(Req("sk-tomaris:live:abc123")), "azizbek")
check("limit survives a colon-bearing key", auth.limit_for("azizbek"), 200)
try:
    auth.authorize(Req("sk-tomaris"))
    check("truncated prefix rejected", "accepted", "401")
except KeyAuthError as e:
    check("truncated prefix rejected", e.status, 401)

path = state_path()
auth = key_auth("azizbek:t7-prod-AAA:200,azizbek:t7-test-BBB:50", path)
for _ in range(50):
    auth.admit(Req("t7-prod-AAA"))
check("a second key for one customer is not pre-spent",
      auth.admit(Req("t7-test-BBB")), "azizbek#2")
check("first key's counter is its own", auth.usage_for("azizbek")["requests"], 50)
check("second key reports its own limit",
      auth.usage_for("azizbek#2")["daily_limit"], 50)

# the cap stays exact when a bot retries concurrently
auth = key_auth("azizbek:t7-C:25")
admitted = []
lock = threading.Lock()


def hammer():
    for _ in range(20):
        try:
            auth.admit(Req("t7-C"))
            with lock:
                admitted.append(1)
        except KeyAuthError:
            pass


threads = [threading.Thread(target=hammer) for _ in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
check("160 concurrent attempts, cap 25", len(admitted), 25)


print("\n8. a hand-edited usage file cannot break every request")
# Raising more headroom mid-day means editing this JSON by hand; a slightly
# wrong shape used to raise inside _bucket() on every request, while /health
# stayed green so nothing alerted.
for broken in [
    {"days": {"azizbek": {"2026-08-20": 0}}},
    {"days": {"azizbek": "7"}},
    {"days": {"azizbek": {"2026-08-20": {"requests": "7"}}}},
    {"days": {"azizbek": {"2026-08-20": {"requests": -5}}}},
    {"days": []},
]:
    path = state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(broken, f)
    auth = key_auth("azizbek:t7-A:200", path)
    try:
        check(f"survives {str(broken)[:40]}", auth.admit(Req("t7-A")), "azizbek")
    except Exception as e:                       # noqa: BLE001 - that is the bug
        check(f"survives {str(broken)[:40]}", f"{type(e).__name__}: {e}", "azizbek")


print("\n9. k -> g softening keeps the wage path alive")
# Uzbek mandatorily softens final -k before a vowel suffix, so "my salary" is
# always "oyligim" -- which matched no cue, no synonym and no code boost.
for question in [
    "oyligimni bermayapti",
    "oyligim kechikdi",
    "oyligini bermadi",
    "ish haqimni bermadi",
    "boshligim 3 oydan beri oyligimni bermayapti",
]:
    check(f"legal cue: {question[:40]}", has_legal_cue(question), True)
    check(f"  Mehnat boost: {question[:40]}",
          match_code_slugs(question), ["mehnat_kodeksi"])
    check(f"  fans out to several queries: {question[:34]}",
          len(queries_for(question, expander)) > 1, True)


print("\n10. tax genitive, assault and divorce reach the right retrieval")
# "daromad soligʻi" folds to "soligi", which does not contain "soliq" -- the
# post-filter dropped 100% of the Tax Code from questions about a named tax.
tax_keys = [("soliq_kodeksi", "387"), ("soliq_kodeksi", "388"),
            ("mehnat_kodeksi", "253")]
for question in ["daromad solig'i qancha foiz", "mol-mulk solig'ini kim to'laydi",
                 "yer solig'i stavkasi qancha", "soliq to'lamadim nima bo'ladi"]:
    kept = exclude_unmentioned_soliq(list(tax_keys), queries_for(question, expander))
    check(f"Tax Code kept: {question[:38]}",
          sum(1 for k in kept if k[0] == "soliq_kodeksi"), 2)
    # ...and the question has to reach retrieval in the first place
    check(f"  legal cue: {question[:38]}", has_legal_cue(question), True)
    check(f"  Soliq boost: {question[:38]}",
          match_code_slugs(question), ["soliq_kodeksi"])
check("wage story still excludes the Tax Code",
      [k for k in exclude_unmentioned_soliq(list(tax_keys),
                                            queries_for("oyligimni bermayapti", expander))],
      [("mehnat_kodeksi", "253")])

# "erim meni uradi" is the canonical example and matched no template: only the
# -ish/-ib forms were listed, never the present or 3rd-person past.
for question in ["erim meni uradi", "erim meni urdi", "erim meni uryapti",
                 "meni kaltakladi"]:
    check(f"assault template: {question}", len(issue_queries(question)) >= 1, True)
for question in ["ajrashmoqchiman", "xotinim bilan ajrashmoqchiman",
                 "ajrashish tartibi", "развестись хочу"]:
    check(f"divorce template: {question}", len(issue_queries(question)) >= 1, True)

# and none of this drags small talk into the corpus
for question in ["Urganchda yashayman", "urug' ekdim", "urush haqida kitob",
                 "ajratib qo'ydim", "Toshkentda havo qanday?", "futbol o'ynadim"]:
    check(f"still small talk: {question}", match_code_slugs(question), [])


print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all customer-readiness tests pass")
