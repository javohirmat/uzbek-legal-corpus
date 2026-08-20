"""Deterministic layer: ground truth about which articles exist.

Zero LLM calls happen in this file. Everything here is exact lookup against the
7,368 audited articles, which is what makes "invented article numbers" and
"silently answering about a repealed article" structurally impossible rather
than merely discouraged by a prompt.
"""
import json
import re
import unicodedata

SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
       "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}

# Okina / apostrophe variants are DELETED (not mapped to ') so that the corpus
# spelling and the slug spelling collapse to the same token:
#   "maʼmuriy" -> "mamuriy"   (slug: mamuriy_javobgarlik_kodeksi)
#   "qogʻozlar" -> "qogozlar" (slug: qimmatli_qogozlar_bozori_qonuni)
_APOS = "ʻʼ‘’ʹʺ`´'"

# Uzbek-first-letter abbreviations that collide across 25 codes are never
# guessed -- they resolve to a clarify question instead. Real words that happen
# to look like abbreviations are dropped outright.
ABBR_STOPLIST = {"ok", "bu", "va", "uz", "yo", "shu", "har"}

# Superscripts run to two digits in this corpus (419²⁰-modda, 145 such articles),
# so the run is matched greedily, not as a single character. The dotted part
# form ("141.2-modda") is accepted too: it is how people type 141² on a Latin
# keyboard, and "141.2" is already the canonical article_id.
_ART_RE = re.compile(
    # up to 6 digits: base articles reach 1199, and a flattened two-digit
    # superscript reaches "41920" (= 419²⁰)
    r"(?<![\d.])(\d{1,6})(?:\.(\d{1,2}))?\s*([¹²³⁴⁵⁶⁷⁸⁹][⁰¹²³⁴⁵⁶⁷⁸⁹]*)?"
    r"\s*[-‐‑‒–—]?\s*modda\w*",
    re.IGNORECASE,
)

# A bare number followed by a count/time word is a quantity, not an article:
# "Mehnat kodeksi bo'yicha 3 oydan beri oylik bermayapti" is not Article 3.
# No leading ^: this is used as .match(tail, num_end), where ^ would only
# anchor at the string's true start, not at pos.
#
# Uzbek is agglutinative, so a fixed list of inflected forms can never be
# complete: "3 oydan" was covered but "3 oyga"/"3 oygacha"/"3 oylik" were not,
# and each miss handed the user a phantom article instead of retrieval. These
# stems therefore take an open suffix. Short stems whose prefix collides with
# ordinary words ("ta" inside "tartibi", "bet" inside "betakror") keep an
# explicit form list instead.
_COUNT_STEMS_OPEN = (
    "oy|yil|kun|hafta|soat|daqiqa|yosh|marta|nafar|kishi|dona|"
    "foiz|protsent|som|sum|dollar|yevro|rubl|million|mln|mlrd|ming|yuz|"
    "baravar|barobar|barobardan|farzand|bola|qavat|xona|sinf|kurs|shaxs|"
    "qism|band|punkt|karra|"
    "yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr"
)
_COUNT_STEMS_EXACT = (
    "ta|tasi|tadan|tada|taga|talik|tacha|"
    "bet|betdan|xil|xildan|xilda"
)
_COUNT_STEMS_RU = (
    "месяц(?:а|ев|у|ам)?|год(?:а|у|ам)?|лет|час(?:а|ов|ам)?|"
    "день|дня|дней|недел(?:я|и|ю|ь)|человек(?:а|ам)?|процент(?:а|ов)?|"
    "рубл(?:ь|я|ей)|доллар(?:а|ов)?|тысяч(?:а|и|у)?|миллион(?:а|ов)?"
)
# The same Russian words after transliteration. pipeline._refs retries a failed
# parse against normalize_query(text), where "3 месяца" has already become
# "3 mesyatsa" -- so a Cyrillic-only guard passed on the raw text and then let
# the quantity through on the retry. This is the form the server actually sees
# for a Russian story that names an Uzbek code.
_COUNT_STEMS_RU_LAT = (
    "mesyats|god|let|nedel|chas|den|dnya|dney|chelovek|tisyach|"
    "raz|shtuk|detey|reben|rebyon"
)
_COUNT_AFTER = re.compile(
    r"\s*[-‐‑‒–—]?\s*(?:"
    r"(?:" + _COUNT_STEMS_OPEN + r")[a-zʻʼ]*"
    r"|(?:" + _COUNT_STEMS_RU_LAT + r")[a-z]*"
    r"|(?:" + _COUNT_STEMS_EXACT + r")"
    r"|(?:" + _COUNT_STEMS_RU + r")"
    r")\b"
)
# After a code name, these tails are story facts, not article ids:
# leftover phone digits, %, the year of 12.05.2024, 12/05 dates,
# and grouped money ("1 500 000") — but not "JK 169 170" (one 3-digit follow-on).
_NOT_ARTICLE_TAIL = re.compile(
    r"(?:\d|%|\.\d{2,4}\b|/\d|\s+\d{3}(?:\s+\d{3})+|\s+\d{3}\s*(?:som|sum|ming|mln))"
)
# Russian cites the article before the code: "ст. 169 УК", "статья 253 ТК".
# Anchored to the end of the look-back window so the number must sit directly
# in front of the code mention, and gated on an explicit citation word so a
# bare quantity ("3 oydan beri mehnat kodeksi") can never match.
_CITE_BEFORE = re.compile(
    r"(?<!\w)(?:st|statya|statyi|statyu|statyasi|modda|moddasi|"
    r"ст|стат[ья][яию]?)\.?\s*"
    r"(\d{1,4})(?:\.(\d{1,2}))?\s*([¹²³⁴⁵⁶⁷⁸⁹][⁰¹²³⁴⁵⁶⁷⁸⁹]*)?[\s,.-]*$",
    re.IGNORECASE,
)

_TITLE_PREFIX = re.compile(r"^ozbekiston respublikasining\s+", re.IGNORECASE)


def norm(s: str) -> str:
    """Fold a string for matching: strip okinas, lowercase, squeeze whitespace."""
    s = unicodedata.normalize("NFC", s)
    for ch in _APOS:
        s = s.replace(ch, "")
    s = s.replace("“", " ").replace("”", " ").replace("«", " ").replace("»", " ")
    return re.sub(r"\s+", " ", s.lower()).strip()


def candidates(digits: str, sup: str | None, part: str | None = None) -> list[str]:
    """Canonical article_id candidates for a parsed reference.

    '480' + '¹' -> ['480.1']      (explicit superscript: unambiguous)
    '141' + '2' -> ['141.2']      (dotted part form == superscript article)
    '480'       -> ['480']        (plain base article)
    '4801'      -> ['4801']       (resolver also tries the flattened map, where
                                   article_raw '4801-modda' points at 480.1)
    """
    if part:
        return [f"{digits}.{part}"]
    if sup:
        return [f'{digits}.{"".join(SUP[c] for c in sup)}']
    return [digits]


class CorpusIndex:
    def __init__(self, records):
        # Front/end matter is structural text, not a citable article. It also
        # carries non-numeric ids ("_front"), so it must never reach int().
        self.articles = [r for r in records if re.fullmatch(r"\d+(\.\d+)?", r["article_id"])]

        self.by_key = {(r["code"], r["article_id"]): r for r in self.articles}
        self.by_flat = {}      # (code, "4801") -> 480¹ record, from article_raw
        self.max_by_slug = {}  # code -> highest base article number
        self.num_to_slugs = {}  # "149" -> {codes that have it}
        for r in self.articles:
            code = r["code"]
            base = int(r["article_id"].split(".")[0])
            self.max_by_slug[code] = max(self.max_by_slug.get(code, 0), base)
            self.num_to_slugs.setdefault(r["article_id"], set()).add(code)
            flat = re.sub(r"\D", "", r.get("article_raw") or "")
            if flat:
                self.by_flat.setdefault((code, flat), r)

        self.slugs = sorted({r["code"] for r in self.articles})
        self._build_aliases(records)
        self._compile_matchers()

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    # aliases: how a human names a code -> which index slugs it covers
    # ------------------------------------------------------------------
    def _build_aliases(self, records):
        titles = {r["code"]: r.get("code_title", "") for r in records}

        groups = {}  # logical code -> [slugs]   (FK part 1 + part 2 merge into one)
        for s in self.slugs:
            groups.setdefault(re.sub(r"[_\-]\d+qism$", "", s), []).append(s)

        # human-facing names
        self.display = {}     # slug -> "Bojxona kodeksi (birinchi qism)"
        self.group_name = {}  # slug -> "Fuqarolik kodeksi"
        for logical, members in groups.items():
            pretty = " ".join(re.split(r"[_\-]+", logical)).capitalize()
            for m in members:
                self.group_name[m] = pretty
                short = _TITLE_PREFIX.sub("", norm(titles.get(m, "")))
                self.display[m] = (
                    titles.get(m, "").split("Respublikasining", 1)[-1].strip()
                    if "Respublikasining" in titles.get(m, "")
                    else (short.capitalize() or pretty)
                )

        self.aliases = {}   # normalized alias -> [slugs]
        self.ambiguous = {}  # normalized alias -> [slugs]  (never auto-resolved)

        def put(alias, members):
            alias = norm(alias)
            if not alias or len(alias) < 2:
                return
            self.aliases.setdefault(alias, [])
            for m in members:
                if m not in self.aliases[alias]:
                    self.aliases[alias].append(m)

        for logical, members in groups.items():
            put(" ".join(re.split(r"[_\-]+", logical)), members)          # "jinoyat kodeksi"
            for m in members:
                t = norm(titles.get(m, ""))
                t = _TITLE_PREFIX.sub("", t)
                t = re.sub(r"\(.*?\)", " ", t).strip()
                if t:
                    put(t, members)                                        # full official title
                    # Titles read "<subject> togrisidagi qonuni/kodeksi", but people
                    # say "Raqobat qonuni" / "Maʼmuriy sud ishlarini yuritish
                    # kodeksi" -- register the short forms too. This also covers
                    # slugs whose spelling drifts from the official title
                    # (mamuriy_sud_ishlarni_... vs "...ishlarini...").
                    subj = re.split(r"\s+togrisida", t)[0].strip()
                    if subj and subj != t:
                        put(subj + " togrisida", members)
                        put(subj + " qonuni", members)
                        put(subj + " kodeksi", members)

        # abbreviations, only when unambiguous
        abbr_owner = {}
        for logical in groups:
            words = re.split(r"[_\-]+", logical)
            abbr_owner.setdefault("".join(w[0] for w in words), []).append(logical)
        for abbr, owners in abbr_owner.items():
            if len(abbr) < 2 or abbr in ABBR_STOPLIST:
                continue
            if len(owners) == 1:
                put(abbr, groups[owners[0]])
            else:
                self.ambiguous[abbr] = [s for o in owners for s in groups[o]]

        # Russian-style names arrive after transliteration ("УК 173" -> "uk
        # 173"). The partner's audience writes Russian, so both the short
        # abbreviations and the spelled-out code names are registered.
        # Deliberately absent: "sk" (Семейный) and "zhk"/"jk" (Жилищный) --
        # both collide with Uzbek abbreviations that mean something else.
        for abbr, members in (("uk", ["jinoyat_kodeksi"]),
                              ("tk", ["mehnat_kodeksi"]),
                              ("gk", ["fuqarolik_kodeksi"]),
                              ("nk", ["soliq_kodeksi"]),
                              ("upk", ["jinoyat_protsessual_kodeksi"]),
                              ("gpk", ["fuqarolik_protsessual_kodeksi"]),
                              ("koao", ["mamuriy_javobgarlik_kodeksi"]),
                              ("ugolovniy kodeks", ["jinoyat_kodeksi"]),
                              ("ugolovnogo kodeksa", ["jinoyat_kodeksi"]),
                              ("trudovoy kodeks", ["mehnat_kodeksi"]),
                              ("trudovogo kodeksa", ["mehnat_kodeksi"]),
                              ("grajdanskiy kodeks", ["fuqarolik_kodeksi"]),
                              ("grajdanskogo kodeksa", ["fuqarolik_kodeksi"]),
                              ("semeyniy kodeks", ["oila_kodeksi"]),
                              ("semeynogo kodeksa", ["oila_kodeksi"]),
                              ("nalogoviy kodeks", ["soliq_kodeksi"]),
                              ("nalogovogo kodeksa", ["soliq_kodeksi"])):
            if abbr in self.aliases or abbr in self.ambiguous:
                continue
            present = [m for m in members if m in groups]
            if present:
                put(abbr, [s for m in present for s in groups[m]])

        # Unique single-word subjects, so "Mehnat 253-modda" resolves without
        # the word "kodeksi". Only when exactly one logical code claims the
        # head word (jinoyat/fuqarolik/mamuriy each head several -> skipped),
        # and never for words that are everyday nouns on their own ("havo",
        # "budjet" -- the weather and the household budget are not statutes).
        head_owners = {}
        for logical in groups:
            head = re.split(r"[_\-]+", logical)[0]
            head_owners.setdefault(head, []).append(logical)
        for head, owners in head_owners.items():
            if len(owners) == 1 and len(head) > 3 and head not in ("havo", "budjet"):
                put(head, groups[owners[0]])

    def _compile_matchers(self):
        self._matchers = []
        for alias in sorted(self.aliases, key=len, reverse=True):
            self._matchers.append((self._pattern(alias), self.aliases[alias], False))
        for alias in sorted(self.ambiguous, key=len, reverse=True):
            self._matchers.append((self._pattern(alias), self.ambiguous[alias], True))

        # Typo-tolerant matching: users type "fuqoro kodeqs" for "fuqarolik
        # kodeksi", and refusing to recognise the code means refusing to answer
        # about a real article. Each word is reduced to a 3-letter prefix.
        #
        # Hyphens split words like spaces, otherwise "jinoyat-protsessual
        # kodeksi" reduces to the same two-token pattern as "jinoyat kodeksi"
        # and a Criminal Code question resolves to the Criminal Procedure Code.
        # Any prefix pattern that still spans more than one logical code is
        # dropped rather than guessed.
        loose = {}
        for alias, members in self.aliases.items():
            words = [w for w in re.split(r"[\s\-]+", alias) if w]
            if len(words) < 2:
                continue
            loose.setdefault(tuple(w[:3] for w in words), set()).update(members)

        self._loose = []
        for key, members in sorted(loose.items(), key=lambda kv: -len(kv[0])):
            logical = {re.sub(r"[_\-]\d+qism$", "", m) for m in members}
            if len(logical) > 1:
                continue
            body = r"[\s\-]+".join(re.escape(k) + r"\w*" for k in key)
            self._loose.append((re.compile(r"(?<!\w)" + body), sorted(members), False))

    @staticmethod
    def _pattern(alias):
        words = alias.split(" ")
        if len(words) == 1 and len(alias) <= 5 and alias.isalpha():
            # short abbreviation: exact token, no suffixes ("jk", "mjk")
            return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)")
        # full name: allow Uzbek case suffixes on the final word
        # ("jinoyat kodeksining", "konstitutsiyadagi")
        body = r"[\s\-]+".join(re.escape(w) for w in words)
        return re.compile(r"(?<!\w)" + body + r"\w*")

    def _mentions(self, text_norm):
        """[(position, slugs, is_ambiguous)] with longest-alias-wins overlap suppression."""
        hits, taken = [], []
        for rx, members, amb in self._matchers:
            for m in rx.finditer(text_norm):
                if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                hits.append((m.start(), members, amb))
        # Misspelled codes are matched too, but only over text no exact alias
        # claimed -- "konstitutsiya 108 ... fuqoro kodeqs 674" must resolve
        # both, not just the correctly spelled one.
        for rx, members, amb in self._loose:
            for m in rx.finditer(text_norm):
                if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                hits.append((m.start(), members, amb))
        return hits

    def find_codes(self, text):
        hits = self._mentions(norm(text))
        return hits[0][1] if hits else None

    # ------------------------------------------------------------------
    # reference parsing + resolution
    # ------------------------------------------------------------------
    def parse_references(self, text, context=""):
        """Extract every 'N-modda' and bind each to the nearest named code.

        `context` is earlier conversation text, used only when the current
        message names no code at all ("...va 12-moddasi-chi?").
        """
        t = norm(text)
        # Glued citations: "JK173"/"jk173" must read as "jk 173". Only the
        # letter->digit seam is split, so "1412modda" and "4801" stay intact.
        t = re.sub(r"(?<=[a-z])(?=\d)", " ", t)
        mentions = self._mentions(t)
        if not mentions and context:
            # fall back to the most recent code named earlier in the thread
            ctx = self._mentions(norm(context))
            if ctx:
                last = max(ctx, key=lambda h: h[0])
                mentions = [(0, last[1], last[2])]
        refs = []
        for m in _ART_RE.finditer(t):
            # Uzbek legal citations name the code BEFORE the number
            # ("Jinoyat kodeksining 173-moddasi"), so bind to the closest
            # preceding mention; only look forward when nothing precedes.
            before = [h for h in mentions if h[0] <= m.start()]
            pool = before or mentions
            if pool:
                _, slugs, amb = min(pool, key=lambda h: abs(h[0] - m.start()))
            else:
                slugs, amb = None, False
            refs.append(
                {
                    "slugs": slugs,
                    "ambiguous": amb,
                    "cands": candidates(m.group(1), m.group(3), m.group(2)),
                    "digits": m.group(1),
                    "raw": m.group(0),
                    "span": m.span(),
                }
            )

        # People also write the number bare: "fuqarolik kodeksi 674 kerak".
        # Only accept a number that directly follows a code mention, so stray
        # figures elsewhere in a sentence are never mistaken for articles.
        # A number glued to a count/time word ("3 oydan", "5 kishi") is a
        # quantity, never an article -- rejecting it keeps situation stories
        # out of the article-lookup path.
        covered = [r["span"] for r in refs]
        for pos, slugs, amb in mentions:
            # Russian names the article BEFORE the code ("ст. 169 УК"), so also
            # look back -- but only when an explicit citation word introduces
            # the number, otherwise "3 oydan beri mehnat kodeksi" would read
            # its quantity as an article.
            head = t[max(0, pos - 40):pos]
            hm = _CITE_BEFORE.search(head)
            if hm and not any(s <= max(0, pos - 40) + hm.start(1) < e for s, e in covered):
                hstart = max(0, pos - 40) + hm.start(1)
                refs.append({
                    "slugs": slugs, "ambiguous": amb,
                    "cands": candidates(hm.group(1), hm.group(3), hm.group(2)),
                    "digits": hm.group(1), "raw": hm.group(1),
                    "span": (hstart, hstart + len(hm.group(1))),
                })
                covered.append((hstart, hstart + len(hm.group(1))))
                continue
            tail = t[pos:pos + 120]
            for m in re.finditer(r"(?<!\w)(\d{1,4})(?:\.(\d{1,2}))?\s*([¹²³⁴⁵⁶⁷⁸⁹][⁰¹²³⁴⁵⁶⁷⁸⁹]*)?(?!\s*-?\s*modda)", tail):
                start = pos + m.start()
                if any(s <= start < e for s, e in covered):
                    continue
                between = t[pos:start]
                if len(between) > 40 or re.search(r"\d", between):
                    continue
                # Anchor the guards to the end of the NUMBER, not to the end of
                # the match: the regex's \s* before the optional superscript
                # swallows the separating space, which made _NOT_ARTICLE_TAIL's
                # leading \d fire on the next article ("JK 169 170" -> nothing).
                num_end = m.end(3) if m.group(3) else (
                    m.end(2) if m.group(2) else m.end(1))
                if _COUNT_AFTER.match(tail, num_end) or _NOT_ARTICLE_TAIL.match(tail, num_end):
                    continue
                refs.append({
                    "slugs": slugs, "ambiguous": amb,
                    "cands": candidates(m.group(1), m.group(3), m.group(2)),
                    "digits": m.group(1), "raw": m.group(0),
                    "span": (start, pos + m.end()),
                })
                covered.append((start, pos + m.end()))
                break
        return refs

    def resolve(self, ref):
        """(status, hint, record).

        EXISTS         -> record is the verbatim article
        REPEALED       -> number is inside the code's range but absent (audited gap)
        OUT_OF_RANGE   -> number is past the code's last article
        AMBIGUOUS_CODE -> abbreviation matches several codes
        NO_CODE        -> no code named anywhere in the request
        """
        if ref["slugs"] is None:
            num = ref["cands"][0]
            return ("NO_CODE", {"num": num, "codes_with": sorted(self.num_to_slugs.get(num, []))}, None)
        if ref["ambiguous"]:
            return ("AMBIGUOUS_CODE", {"num": ref["cands"][0], "slugs": ref["slugs"]}, None)

        for slug in ref["slugs"]:
            for cand in ref["cands"]:
                if (slug, cand) in self.by_key:
                    return ("EXISTS", (slug, cand), self.by_key[(slug, cand)])
        # "4801-modda" typed flat -> article_raw map resolves it to 480¹ exactly.
        #
        # Only for a number the user typed FLAT. When they explicitly asked for
        # an insert ("JK 169⁵", "MJK 27.2"), every candidate carries a dot, and
        # this map is keyed on the bare digits -- so it used to answer with the
        # BASE article (169 Oʻgʻrilik) as if it were the one requested. Serving
        # a different article is worse than saying the number does not exist,
        # which is the whole guarantee behind the deterministic path.
        asked_for_insert = any("." in c for c in ref["cands"])
        if not asked_for_insert:
            for slug in ref["slugs"]:
                rec = self.by_flat.get((slug, ref["digits"]))
                if rec is not None:
                    return ("EXISTS", (slug, rec["article_id"]), rec)

        base = int(ref["digits"])
        top = max(self.max_by_slug.get(s, 0) for s in ref["slugs"])
        hint = {"slugs": ref["slugs"], "num": ref["cands"][0], "max": top}
        return (("REPEALED" if base <= top else "OUT_OF_RANGE"), hint, None)

    # ------------------------------------------------------------------
    # citation verification (semantic path)
    # ------------------------------------------------------------------
    def allowed_ids(self, articles):
        """Article numbers the model may legitimately mention.

        = the retrieved articles, PLUS every article they themselves
        cross-reference. 21% of this corpus cross-references other articles
        ("ushbu Kodeksning 177-moddasida..."), so without this the checker
        would reject faithful quotations as hallucinations.
        """
        ok = set()
        for a in articles:
            ok.add(a["article_id"])
            ok.add(re.sub(r"\D", "", a.get("article_raw") or "") or a["article_id"])
            for m in _ART_RE.finditer(norm(a["text"])):
                ok.update(candidates(m.group(1), m.group(3), m.group(2)))
                ok.add(m.group(1))
        return ok

    def bad_citations(self, answer, allowed):
        """Citations in the answer that point at nothing we supplied."""
        bad = []
        for m in _ART_RE.finditer(norm(answer)):
            cands = set(candidates(m.group(1), m.group(3), m.group(2))) | {m.group(1)}
            if not (cands & allowed):
                bad.append(m.group(0))
        return bad

    def cited_subset(self, answer, articles):
        """Keep only the articles the answer actually cites.

        Retrieval hands the model more candidates than it needs; listing all of
        them as "Manbalar" shows the user irrelevant sources (a Family Code
        article under a labour answer) and reads as sloppy grounding.
        """
        seen = set()
        for m in _ART_RE.finditer(norm(answer)):
            seen.update(candidates(m.group(1), m.group(3), m.group(2)))
            seen.add(m.group(1))
        used = [
            a for a in articles
            if a["article_id"] in seen
            or (re.sub(r"\D", "", a.get("article_raw") or "") in seen)
        ]
        return used or articles

    def name_of(self, slugs):
        return ", ".join(sorted({self.group_name.get(s, s) for s in slugs}))
