"""Bridge everyday Uzbek to the corpus's formal legal wording.

People ask "armiyaga bormasa nima boladi"; the statute says "harbiy xizmatga
chaqiruvdan boʻyin tovlash". Those embed far apart, so the question both looked
non-legal (cosine 0.509, indistinguishable from a cooking question at 0.527) and
retrieved the wrong articles. Appending the formal terms moved it to 0.353 and
surfaced Jinoyat kodeksi 225-modda, the article that actually answers it, while
leaving non-legal questions untouched.

Expansion applies to the retrieval query only -- the model still sees the user's
own words.
"""
import json
import os
import re

from corpus_index import norm


class QueryExpander:
    def __init__(self, rules):
        self.rules = []
        for r in rules:
            pats = []
            for phrase in r.get("when", []):
                words = norm(phrase).split()
                if not words:
                    continue
                body = r"\s+".join(re.escape(w) for w in words)
                pats.append(re.compile(r"(?<!\w)" + body + r"\w*"))
            if pats and r.get("add"):
                self.rules.append((pats, r["add"]))

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path):
            return cls([])
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def expand(self, question):
        q = norm(question)
        extra = [add for pats, add in self.rules if any(p.search(q) for p in pats)]
        return f"{question} {' '.join(extra)}" if extra else question

    def issue_adds(self, question):
        """Each matched rule as its own search string (issue-level, not a paraphrase)."""
        q = norm(question)
        return [add for pats, add in self.rules if any(p.search(q) for p in pats)]

    def __len__(self):
        return len(self.rules)
