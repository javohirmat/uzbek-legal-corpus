#!/usr/bin/env python3
"""parse_articles.py — Uzbek legal corpus: raw txt -> article-level JSONL + manifest.csv

Key conventions handled:
  * tolerant heading regex: "N-modda", "N- modda", "N -modda"
  * lex.uz flattens inserted-article superscripts: 480¹ appears as "4801".
    Article ids are resolved by DOCUMENT POSITION (lookahead walker), never by
    raw value — "261" between 26 and 27 is 26¹ (id 26.1); between 260 and 262
    it is article 261.
  * section (BOʻLIM/QISM) and chapter (bob) context is tracked per article.
  * pre-article front matter and post-article end matter (e.g. the Criminal
    Code's legal-terms glossary) are emitted as _front / _end records.

Usage: python3 scripts/parse_articles.py  (run from repo root)
"""
import csv, hashlib, json, re, sys
from pathlib import Path

RAW = Path("data/raw"); OUT = Path("data/articles"); OUT.mkdir(parents=True, exist_ok=True)
SNAPSHOT = "2026-07"

HEAD = re.compile(r"^\s*(\d+)\s*-\s*modda\b\.?\s*(.*)$")
CHAP = re.compile(r"^\s*\d+\s*-\s*(bob|BOB)\b")
SECT = re.compile(r"^\s*(?:[IVXL]+|BIRINCHI|IKKINCHI|UCHINCHI|TOʻRTINCHI|BESHINCHI|"
                  r"OLTINCHI|YETTINCHI|SAKKIZINCHI|TOʻQQIZINCHI|OʻNINCHI|UMUMIY|MAXSUS)"
                  r"\s+(?:KICHIK\s+)?(BOʻLIM|QISM|boʻlim|qism)\b")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# code -> (official title, lex.uz doc id verified during audit ('' = look up), adopted)
META = {
 "konstitutsiya": ("Oʻzbekiston Respublikasining Konstitutsiyasi", "-6445145", "2023-05-01"),
 "fuqarolik_kodeksi_1qism": ("Oʻzbekiston Respublikasining Fuqarolik kodeksi (birinchi qism)", "-111189", "1995-12-21"),
 "fuqarolik_kodeksi_2qism": ("Oʻzbekiston Respublikasining Fuqarolik kodeksi (ikkinchi qism)", "-180552", "1996-08-29"),
 "fuqarolik_protsessual_kodeksi": ("Oʻzbekiston Respublikasining Fuqarolik protsessual kodeksi", "-3517337", "2018-01-22"),
 "jinoyat_kodeksi": ("Oʻzbekiston Respublikasining Jinoyat kodeksi", "-111453", "1994-09-22"),
 "jinoyat_protsessual_kodeksi": ("Oʻzbekiston Respublikasining Jinoyat-protsessual kodeksi", "-111460", "1994-09-22"),
 "jinoyat-ijroiya_kodeksi": ("Oʻzbekiston Respublikasining Jinoyat-ijroiya kodeksi", "-163629", "1997-04-25"),
 "mamuriy_javobgarlik_kodeksi": ("Oʻzbekiston Respublikasining Maʼmuriy javobgarlik toʻgʻrisidagi kodeksi", "-97664", "1994-09-22"),
 "mamuriy_sud_ishlarni_yuritish_kodeksi": ("Oʻzbekiston Respublikasining Maʼmuriy sud ishlarini yuritish toʻgʻrisidagi kodeksi", "-3527353", "2018-01-25"),
 "iqtisodiy_protsessual_kodeksi": ("Oʻzbekiston Respublikasining Iqtisodiy protsessual kodeksi", "-3523891", "2018-01-24"),
 "mehnat_kodeksi": ("Oʻzbekiston Respublikasining Mehnat kodeksi", "-6257288", "2022-10-28"),
 "oila_kodeksi": ("Oʻzbekiston Respublikasining Oila kodeksi", "-104720", "1998-04-30"),
 "yer_kodeksi": ("Oʻzbekiston Respublikasining Yer kodeksi", "-152653", "1998-04-30"),
 "soliq_kodeksi": ("Oʻzbekiston Respublikasining Soliq kodeksi", "-4674902", "2019-12-30"),
 "budjet_kodeksi": ("Oʻzbekiston Respublikasining Budjet kodeksi", "-2304138", "2013-12-26"),
 "bojxona_kodeksi": ("Oʻzbekiston Respublikasining Bojxona kodeksi", "-2876354", "2016-01-20"),
 "havo_kodeksi": ("Oʻzbekiston Respublikasining Havo kodeksi", "-55594", "1993-05-07"),
 "suv_kodeksi": ("Oʻzbekiston Respublikasining Suv kodeksi", "-7655343", "2025-07-30"),
 "saylov_kodeksi": ("Oʻzbekiston Respublikasining Saylov kodeksi", "-4386848", "2019-06-25"),
 "shaharsozlik_kodeksi": ("Oʻzbekiston Respublikasining Shaharsozlik kodeksi", "-5307951", "2021-02-22"),
 "investitsiyalar_faoliyati_qonuni": ("Oʻzbekiston Respublikasining \u201cInvestitsiyalar va investitsiya faoliyati toʻgʻrisida\u201dgi Qonuni", "-4664142", "2019-12-25"),
 "markaziy_bank_qonuni": ("Oʻzbekiston Respublikasining \u201cOʻzbekiston Respublikasining Markaziy banki toʻgʻrisida\u201dgi Qonuni", "-72266", "2019-11-11"),
 "qimmatli_qogozlar_bozori_qonuni": ("Oʻzbekiston Respublikasining \u201cQimmatli qogʻozlar bozori toʻgʻrisida\u201dgi Qonuni", "-1374865", "2015-06-03"),
 "raqobat_qonuni": ("Oʻzbekiston Respublikasining \u201cRaqobat toʻgʻrisida\u201dgi Qonuni", "-6518381", "2023-07-03"),
 "tadbirkorlik_faoliyati_qonuni": ("Oʻzbekiston Respublikasining \u201cTadbirkorlik faoliyati erkinligining kafolatlari toʻgʻrisida\u201dgi Qonuni", "-2006789", "2012-05-02"),
}

def resolve_ids(nums):
    """Position-aware id assignment. Returns list of (id_str, kind)."""
    out, prev = [], (nums[0] - 1 if nums and nums[0] > 1 else 0)
    for j, n in enumerate(nums):
        nxt = nums[j + 1] if j + 1 < len(nums) else None
        cand = None
        for L in range(1, len(str(n))):
            b = int(str(n)[:L])
            if prev <= b <= prev + 40:
                cand = (b, str(n)[L:]); break
        base_ok = prev < n <= prev + 40
        if base_ok and cand and cand[0] != n:  # ambiguous (e.g. "41" after 4)
            if nxt is not None and (nxt == cand[0] + 1 or str(nxt).startswith(str(cand[0]))):
                out.append((f"{cand[0]}.{int(cand[1])}", "insert")); prev = max(prev, cand[0]); continue
            out.append((str(n), "base")); prev = n; continue
        if base_ok:
            out.append((str(n), "base")); prev = n; continue
        if cand:
            out.append((f"{cand[0]}.{int(cand[1])}", "insert")); prev = max(prev, cand[0]); continue
        out.append((str(n), "unresolved"))
    return out

def display(art_id):
    if "." in art_id:
        b, s = art_id.split(".")
        return f"{b}{s.translate(SUP)}-modda"
    return f"{art_id}-modda"

manifest = []
for path in sorted(RAW.glob("*.txt")):
    code = path.stem
    title_official, lex_doc, adopted = META.get(code, (code, "", ""))
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # pass 1: locate structure
    marks = []  # (line_idx, type, payload)
    for i, l in enumerate(lines):
        m = HEAD.match(l)
        if m:
            marks.append((i, "head", (int(m.group(1)), m.group(2).strip()))); continue
        if SECT.match(l):
            marks.append((i, "sect", l.strip())); continue
        if CHAP.match(l):
            marks.append((i, "chap", l.strip()))
    heads = [(i, p) for i, t, p in marks if t == "head"]
    ids = resolve_ids([n for _, (n, _) in heads])

    recs, sect, chap = [], "", ""
    first_head = heads[0][0]
    front = [l.strip() for l in lines[:first_head] if l.strip()
             and not SECT.match(l) and not CHAP.match(l)]
    # front matter beyond the bare title line
    if len(front) > 1:
        recs.append({"article_id": "_front", "article_display": "front_matter",
                     "title": "", "text": "\n".join(front)})

    mark_ptr = 0
    for k, ((hi, (num, art_title)), (aid, kind)) in enumerate(zip(heads, ids)):
        while mark_ptr < len(marks) and marks[mark_ptr][0] <= hi:
            i, t, p = marks[mark_ptr]
            if t == "sect": sect, chap = p, ""
            elif t == "chap": chap = p
            mark_ptr += 1
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body, cut = [], None
        for i in range(hi + 1, end):
            l = lines[i]
            if k + 1 == len(heads) and SECT.match(l):  # end matter (e.g. JK glossary)
                cut = i; break
            if SECT.match(l) or CHAP.match(l): continue
            if l.strip(): body.append(l.strip())
        recs.append({"article_id": aid, "article_display": display(aid),
                     "article_raw": f"{num}-modda", "kind": kind, "title": art_title,
                     "section": sect, "chapter": chap, "text": "\n".join(body)})
        if cut is not None:
            tail = [l.strip() for l in lines[cut:] if l.strip()]
            recs.append({"article_id": "_end", "article_display": "end_matter",
                         "title": tail[0] if tail else "", "text": "\n".join(tail)})

    outp = OUT / f"{code}.jsonl"
    with outp.open("w", encoding="utf-8") as f:
        for seq, r in enumerate(recs, 1):
            f.write(json.dumps({"seq": seq, "code": code, "code_title": title_official,
                                **r, "lex_uz_doc": lex_doc, "snapshot_date": SNAPSHOT},
                               ensure_ascii=False) + "\n")

    arts = [r for r in recs if not r["article_id"].startswith("_")]
    unres = [r["article_id"] for r in arts if r.get("kind") == "unresolved"]
    manifest.append({"filename": path.name, "code_title": title_official,
                     "lex_uz_doc": lex_doc, "adopted": adopted,
                     "articles_count": len(arts), "last_article": arts[-1]["article_id"],
                     "sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "snapshot_date": SNAPSHOT})
    flag = f"  !! unresolved: {unres}" if unres else ""
    print(f"{code:44} {len(arts):4} articles, last {arts[-1]['article_id']:>6}{flag}")

with open("manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
    w.writeheader(); w.writerows(manifest)
print(f"\n{sum(m['articles_count'] for m in manifest)} articles total -> data/articles/ + manifest.csv")
