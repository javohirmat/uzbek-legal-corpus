#!/usr/bin/env bash
# Five proven curls against a live Tomaris RAG box (prospect demo).
#
#   bash rag/demo_prospect.sh
#   TOMARIS_API=http://host:port bash rag/demo_prospect.sh
#
# Prints retrieval_mode, citations, and the first ~400 chars of content.
# Default URL is the current Vast mapping; override via TOMARIS_API.
# Does not restart services. 90s per request.
set -uo pipefail

API="${TOMARIS_API:-http://140.150.159.1:22879}"
API="${API%/}"
TIMEOUT="${TOMARIS_TIMEOUT:-90}"

json_body() {
  python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}]}, ensure_ascii=False))' "$1"
}

summarize() {
  # args: NAME EXPECTED BODY.json  (heredoc is the printer; body is a file)
  python3 - "$@" <<'PY'
import json, re, sys

name, expect, path = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(path, encoding="utf-8").read()
cyr = re.compile(r"[\u0400-\u04FF]")
suv_re = re.compile(r"suv[_ ]?kodeks", re.I)
fk_re = re.compile(r"fuqarolik|ijara", re.I)
mk_re = re.compile(r"mehnat", re.I)
banned_uz = ("qonun buzilgan", "ikki qonun buzilishi", "siz aybdorsiz")
banned_ru = ("закон нарушен", "вы виновны", "вы будете нести ответственность")

def fail(why):
    print(f"RESULT\t{name}\tFAIL\t{why}")
    sys.exit(2)

if not raw.strip():
    fail("empty body (box down or credit gone?)")

try:
    d = json.loads(raw)
except json.JSONDecodeError:
    fail("not JSON: " + raw[:180].replace("\n", " "))

err = d.get("error")
if err:
    fail(f"api error: {err}")

mode = d.get("retrieval_mode") or ""
cites = d.get("citations") or []
choices = d.get("choices") or []
content = ""
if choices:
    content = (choices[0].get("message") or {}).get("content") or ""
codes = [str(c.get("code") or "") for c in cites]
cite_s = " | ".join(
    f"{c.get('code','')} {c.get('article','')}".strip() for c in cites
) or "(none)"
preview = content.replace("\n", " ").strip()
if len(preview) > 400:
    preview = preview[:400] + "…"

print(f"mode:      {mode or '(missing)'}")
print(f"citations: {cite_s}")
print(f"content:   {preview}")

low = content.lower()
why = "ok"

if expect == "deterministic":
    if mode != "deterministic":
        fail(f"want retrieval_mode=deterministic, got {mode!r}")
    if "999" not in content:
        fail("deterministic reply does not mention 999")
elif expect == "article-lookup":
    if mode != "article-lookup":
        fail(f"want retrieval_mode=article-lookup, got {mode!r}")
    joined = " ".join(codes).lower() + " " + low
    if "konstitutsiya" not in joined and "149" not in content:
        fail("lookup did not cite Konstitutsiya 149")
elif expect == "flood":
    suv_only = bool(codes) and all(suv_re.search(c or "") for c in codes)
    suv_in_text = bool(suv_re.search(low)) and not fk_re.search(low)
    has_fk = any(fk_re.search(c or "") for c in codes) or bool(fk_re.search(low))
    if suv_only or (not has_fk and suv_in_text):
        fail("Suv-only (need Fuqarolik ijara/zarar)")
    if not has_fk:
        fail("no Fuqarolik/ijara in citations or answer")
    why = "not Suv-only"
elif expect == "labour":
    has_mk = any(mk_re.search(c or "") for c in codes) or bool(mk_re.search(low))
    if not has_mk:
        fail("no Mehnat citation")
    if any(p in low for p in banned_uz):
        fail("banned verdict phrasing")
elif expect == "labour-ru":
    has_mk = any(mk_re.search(c or "") for c in codes) or bool(mk_re.search(low))
    if not has_mk:
        fail("no Mehnat citation")
    letters = [ch for ch in content if ch.isalpha()]
    cyr_n = sum(1 for ch in letters if cyr.match(ch))
    if letters and cyr_n / len(letters) < 0.4:
        fail("answer is not mostly Russian")
    blob = content.lower()
    if any(p in blob for p in banned_ru) or any(p in blob for p in banned_uz):
        fail("banned verdict phrasing")
    if "я не юрист" not in blob and "men yurist emasman" not in blob:
        why = "ok (disclaimer missing)"
else:
    fail(f"unknown expect {expect!r}")

print(f"RESULT\t{name}\tPASS\t{why}")
PY
}

ask() {
  local name="$1" expect="$2" question="$3"
  echo
  echo "======== $name ========"
  echo "Q: $question"
  local tmp http body
  tmp="$(mktemp)"
  http="$(
    curl -sS -o "$tmp" -w "%{http_code}" --max-time "$TIMEOUT" \
      -X POST "$API/v1/chat/completions" \
      -H "Content-Type: application/json; charset=utf-8" \
      --data-binary "$(json_body "$question")"
  )" || {
    echo "curl failed (box down or credit gone?)"
    printf 'RESULT\t%s\tFAIL\tcurl error HTTP %s\n' "$name" "${http:-n/a}"
    rm -f "$tmp"
    return 0
  }
  echo "HTTP $http"
  if [ "$http" != "200" ]; then
    echo "body: $(head -c 200 "$tmp")"
    printf 'RESULT\t%s\tFAIL\tHTTP %s\n' "$name" "$http"
    rm -f "$tmp"
    return 0
  fi
  summarize "$name" "$expect" "$tmp" || true
  rm -f "$tmp"
}

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

echo "Tomaris prospect demo  API=$API  timeout=${TIMEOUT}s"
echo "health:"
if ! curl -sS --max-time 10 "$API/health" | tee -a "$LOG"; then
  echo
  echo "FAIL  health check (do not destroy the box; top up Vast credit if this persists)"
  exit 1
fi
echo | tee -a "$LOG"

{
  ask JK999 deterministic "JK 999-modda nima deydi?"
  ask K149 article-lookup "Konstitutsiyaning 149-moddasida nima deyilgan?"
  ask FLOOD flood "Kuchli yomgʻirdan keyin ijaraga turgan kvartiramizni suv bosdi, mebel va texnika shikastlandi. Uy egasi taʼmirni mendan undirmoqchi, men esa uning tomi oqardi deb oʻylayman. Ijara va zarar boʻyicha kim javobgar?"
  ask LABOUR labour "Ish beruvchi ikki oydan beri ish haqini bermayapti va meni ishdan haydab yubordi. Nima qilishim mumkin?"
  ask RU-LABOUR labour-ru "Работодатель два месяца не выплачивает зарплату и уволил меня. Что делать?"
} | tee -a "$LOG"

echo
echo "======== table ========"
printf "%-12s %-6s %s\n" "case" "gate" "note"
fails=0
while IFS=$'\t' read -r _ name gate note; do
  printf "%-12s %-6s %s\n" "$name" "$gate" "$note"
  [ "$gate" = "FAIL" ] && fails=$((fails + 1))
done < <(grep $'^RESULT\t' "$LOG")
echo
if [ "$fails" -gt 0 ]; then
  echo "$fails failed. Do not destroy Vast 47783872; top up credit if the box died mid-run."
  exit 1
fi
echo "all five passed. Do not destroy Vast 47783872 — credit is low; top up."
