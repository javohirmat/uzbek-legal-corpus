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
# so the run is matched greedily, not as a single character.
_ART_RE = re.compile(
    # up to 6 digits: base articles reach 1199, and a flattened two-digit
    # superscript reaches "41920" (= 419²⁰)
    r"(?<![\d.])(\d{1,6})\s*([¹²³⁴⁵⁶⁷⁸⁹][⁰¹²³⁴⁵⁶⁷⁸⁹]*)?\s*[-‐‑‒–—]?\s*modda\w*",
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


def candidates(digits: str, sup: str | None) -> list[str]:
    """Canonical article_id candidates for a parsed reference.

    '480' + '¹' -> ['480.1']      (explicit superscript: unambiguous)
    '480'       -> ['480']        (plain base article)
    '4801'      -> ['4801']       (resolver also tries the flattened map, where
                                   article_raw '4801-modda' points at 480.1)
    """
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

    def _compile_matchers(self):
        self._matchers = []
        for alias in sorted(self.aliases, key=len, reverse=True):
            self._matchers.append((self._pattern(alias), self.aliases[alias], False))
        for alias in sorted(self.ambiguous, key=len, reverse=True):
            self._matchers.append((self._pattern(alias), self.ambiguous[alias], True))

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
                    "cands": candidates(m.group(1), m.group(2)),
                    "digits": m.group(1),
                    "raw": m.group(0),
                }
            )
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
        # "4801-modda" typed flat -> article_raw map resolves it to 480¹ exactly
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
                ok.update(candidates(m.group(1), m.group(2)))
                ok.add(m.group(1))
        return ok

    def bad_citations(self, answer, allowed):
        """Citations in the answer that point at nothing we supplied."""
        bad = []
        for m in _ART_RE.finditer(norm(answer)):
            cands = set(candidates(m.group(1), m.group(2))) | {m.group(1)}
            if not (cands & allowed):
                bad.append(m.group(0))
        return bad

    def name_of(self, slugs):
        return ", ".join(sorted({self.group_name.get(s, s) for s in slugs}))
