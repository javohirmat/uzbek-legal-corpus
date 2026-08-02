"""Customer-configured answers: "if a customer asks X, reply exactly Y".

Checked before retrieval and before the model, so a configured answer is
returned verbatim, every time, with no paraphrasing and no possibility of
hallucination. This is what a bank or agency needs in order to put a chatbot
in front of customers: for the questions they care about, the answer is theirs,
not the model's.

Rules live in a JSON file so a non-engineer can edit them:

    [
      {
        "id": "kredit-foizi",
        "any": ["kredit foizi", "foiz stavkasi"],   # any one of these triggers
        "all": ["isteʼmol"],                        # and all of these must appear
        "not": ["ipoteka"],                         # but none of these
        "answer": "Isteʼmol krediti — yillik 24%.",
        "source": "Bank tariflari (2026-08)"
      }
    ]

Matching is okina/apostrophe-insensitive, so "isteʼmol", "iste'mol" and
"istemol" all match the same rule.
"""
import json
import os
import re

from corpus_index import norm


def _phrase(p):
    """Word-boundary matcher for a trigger phrase.

    Plain substring matching silently crosses word boundaries: the phrase
    "ish tartibi" occurs inside "berish tartibi", so a land-lease question
    ("yer uchastkasini berish tartibi") matched an office-hours rule. Anchor
    both ends, and allow any whitespace between words.
    """
    words = norm(p).split()
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)")


class Overrides:
    def __init__(self, rules):
        self.rules = []
        for r in rules:
            self.rules.append({
                "id": r.get("id", ""),
                "any": [p for p in map(_phrase, r.get("any", [])) if p],
                "all": [p for p in map(_phrase, r.get("all", [])) if p],
                "not": [p for p in map(_phrase, r.get("not", [])) if p],
                "answer": r["answer"],
                "source": r.get("source", ""),
            })

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path):
            return cls([])
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def match(self, question):
        """First matching rule wins; rule order is the priority order."""
        q = norm(question)
        for r in self.rules:
            if r["any"] and not any(p.search(q) for p in r["any"]):
                continue
            if r["all"] and not all(p.search(q) for p in r["all"]):
                continue
            if r["not"] and any(p.search(q) for p in r["not"]):
                continue
            return r
        return None

    def __len__(self):
        return len(self.rules)
