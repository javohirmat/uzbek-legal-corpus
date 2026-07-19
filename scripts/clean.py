#!/usr/bin/env python3
"""clean.py — the normalization pipeline applied to lex.uz statute text.

Steps (in order), reflecting the published corpus:
 1. drop BOM (U+FEFF) and CR line endings
 2. remove lex.uz amendment annotations: lines whose first non-space char is "("
    AND that contain a gazette marker (milliy bazasi / QHT / Axborotnomasi).
    Inline legal parentheticals like "(mansabdor shaxs)" are never touched.
 3. remove lex.uz UI cruft lines (PDF-faylidagi rasmiy manba, tahrirga qarang, …)
 4. okina/apostrophe canonicalization: U+2018 -> U+02BB (okina in oʻ/gʻ);
    U+2019 -> U+02BC (tutuq belgisi) — safe because in these texts U+2019 only
    occurs after a letter, never as a closing quote
 5. strip stray control chars (e.g. U+001E)
 6. normalize spaced article headings: "599- modda" / "1843 -modda" -> "N-modda"
 7. (constitution-style texts) split headings glued to chapter lines or to the
    previous sentence onto their own line; re-join hyphen-wrapped compounds

Usage: python3 scripts/clean.py in.txt out.txt [--split-glued-headings]
"""
import re, sys

GAZETTE = re.compile(r"milliy bazasi|QHT|Axborotnomasi")
UI = re.compile(r"PDF-faylidagi rasmiy manba|tahrirga qarang|Ҳужжат|тинглаш|Аудиони")

def clean(text: str, split_glued: bool = False) -> str:
    text = text.replace("\ufeff", "").replace("\r", "")
    kept = []
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith("(") and GAZETTE.search(line):
            continue                       # amendment annotation
        if UI.search(line):
            continue                       # site cruft
        kept.append(line)
    text = "\n".join(kept)
    text = text.replace("\u2018", "\u02bb").replace("\u2019", "\u02bc")
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    text = re.sub(r"^(\s*)(\d+)\s*-\s*modda", r"\1\2-modda", text, flags=re.M)
    if split_glued:
        text = re.sub(r"((?:bob|BOB|boʻlim|BOʻLIM)\.[^\n]*?)\s+(\d+-modda\.)", r"\1\n\2", text)
        text = re.sub(r"([^\n]{5,}?[.!?»])\s+(\d+-modda\.)", r"\1\n\2", text)
        text = re.sub(r"^(\s*\d+-modda\.)\s+(\S)", r"\1\n\2", text, flags=re.M)
        text = re.sub(r"([\w\u02bb\u02bc])-\n\s*([a-zа-я\u02bb])", r"\1-\2", text)
    return text

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = clean(open(src, encoding="utf-8").read(), "--split-glued-headings" in sys.argv)
    open(dst, "w", encoding="utf-8").write(out)
    print(f"cleaned -> {dst}")
