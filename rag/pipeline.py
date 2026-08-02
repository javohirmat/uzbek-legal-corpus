"""The pipeline: deterministic resolution -> grounded generation -> citation audit.

Order matters. Anything the deterministic layer can answer exactly (an article
number that exists, or provably does not) never reaches the model, so those
answers cannot be hallucinated. Only open questions go to generation, and what
comes back is audited against the articles that were actually supplied.
"""
import json

import dspy

import config as C
from corpus_index import CorpusIndex
from answer_rules import Overrides
from retriever import Retriever

lm = dspy.LM(
    f"openai/{C.VLLM_MODEL}",
    api_base=C.VLLM_BASE,
    api_key=C.VLLM_KEY,
    temperature=C.TEMPERATURE,
    max_tokens=C.MAX_TOKENS,
    extra_body={"chat_template_kwargs": {"enable_thinking": C.ENABLE_THINKING}},
)
dspy.configure(lm=lm)

SYSTEM = (
    "Sen Oʻzbekiston qonunchiligi boʻyicha yordamchisan. Javobni FAQAT quyida "
    "berilgan moddalar matni asosida yoz. Har bir daʼvodan keyin (Kodeks nomi, "
    "N-modda) koʻrinishida iqtibos keltir. Agar berilgan moddalarda javob "
    "boʻlmasa, «Berilgan moddalarda bunga javob yoʻq» deb yoz. Berilmagan modda "
    "raqamini hech qachon oʻylab topma."
)


class GroundedAnswer(dspy.Signature):
    """Sen Oʻzbekiston qonunchiligi boʻyicha yordamchisan. Javobni FAQAT berilgan
    moddalar matni asosida yoz. Har bir daʼvodan keyin (Kodeks nomi, N-modda)
    koʻrinishida iqtibos keltir. Berilgan moddalarda javob boʻlmasa —
    «Berilgan moddalarda bunga javob yoʻq» deb yoz. Berilmagan modda raqamini
    hech qachon oʻylab topma."""

    question = dspy.InputField(desc="foydalanuvchi savoli")
    articles = dspy.InputField(desc="tegishli qonun moddalari (toʻliq matn)")
    answer = dspy.OutputField(desc="iqtiboslangan javob (oʻzbek tilida)")


def _format(articles):
    return "\n\n".join(
        f'[{a["code_title"]} | {a["article_display"]}'
        + (f' | {a["title"]}' if a["title"] else "")
        + f']\n{a["text"][: C.MAX_ARTICLE_CHARS]}'
        for a in articles
    )


class UpstreamUnavailable(RuntimeError):
    """vLLM is down, restarting, or otherwise unreachable."""


def _sources(articles):
    seen, out = set(), []
    for a in articles:
        key = (a["code_title"], a["article_display"])
        if key not in seen:
            seen.add(key)
            out.append(f'— {a["code_title"]}, {a["article_display"]}')
    return "\n".join(out)


class LegalRAG(dspy.Module):
    def __init__(self, index_path=C.INDEX_JSON):
        super().__init__()
        self.index = CorpusIndex.load(index_path)
        self.retriever = Retriever(self.index.articles)
        self.overrides = Overrides.load(C.OVERRIDES_JSON)
        self.generate = dspy.Predict(GroundedAnswer)

    # ---------------- deterministic messages (no LLM) ----------------
    def _missing(self, status, hint):
        if status == "NO_CODE":
            names = sorted({self.index.group_name[s] for s in hint["codes_with"]})
            if not names:
                return f'{hint["num"]}-modda bazadagi hech bir hujjatda topilmadi.'
            shown = ", ".join(names[:5]) + (f" va yana {len(names) - 5} ta hujjat" if len(names) > 5 else "")
            return (
                f'{hint["num"]}-modda bir nechta hujjatda mavjud ({shown}). '
                "Qaysi kodeks yoki qonun haqida soʻrayotganingizni aniqlashtiring."
            )
        if status == "AMBIGUOUS_CODE":
            names = ", ".join(sorted({self.index.group_name[s] for s in hint["slugs"]}))
            return (
                f'Qisqartma bir nechta hujjatga toʻgʻri keladi ({names}). '
                "Hujjat nomini toʻliq yozing."
            )
        names = self.index.name_of(hint["slugs"])
        if status == "REPEALED":
            return (
                f'{names}da {hint["num"]}-modda mavjud emas — u bekor qilingan '
                "yoki hech qachon boʻlmagan."
            )
        return f'{names}da {hint["num"]}-modda yoʻq. Oxirgi modda — {hint["max"]}-modda.'

    # ---------------- generation + citation audit ----------------
    def _raw_call(self, question, ctx):
        """Fallback when DSPy cannot parse the structured reply (fine-tuned
        models do not always honour field markers)."""
        out = lm(messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"MODDALAR:\n{ctx}\n\nSAVOL: {question}"},
        ])
        return out[0] if isinstance(out, list) else str(out)

    def _grounded(self, question, articles):
        ctx = _format(articles)
        allowed = self.index.allowed_ids(articles)
        try:
            answer = self.generate(question=question, articles=ctx).answer
        except Exception as first:
            # A structured-output parse failure is recoverable via a plain call;
            # an unreachable vLLM is not, and must not surface as a raw traceback.
            try:
                answer = self._raw_call(question, ctx)
            except Exception as second:
                raise UpstreamUnavailable(str(second)) from first

        bad = self.index.bad_citations(answer, allowed)
        if bad:
            retry = (
                f"{question}\n\n[DIQQAT: faqat berilgan moddalarga iqtibos qil. "
                f'Quyidagilar berilmagan: {", ".join(sorted(set(bad)))}]'
            )
            try:
                answer = self.generate(question=retry, articles=ctx).answer
            except Exception as first:
                try:
                    answer = self._raw_call(retry, ctx)
                except Exception as second:
                    raise UpstreamUnavailable(str(second)) from first
            if self.index.bad_citations(answer, allowed):
                return (
                    "Ishonchli javob berish uchun bazadagi maʼlumot yetarli emas. "
                    "Savolni aniqroq yozing yoki modda raqamini koʻrsating.",
                    False,
                )
        return answer, True

    # ---------------- entry point ----------------
    def forward(self, question, context=""):
        # Customer-configured answers outrank everything, including the corpus:
        # if a bank has specified the reply to a question, that reply is the
        # answer -- verbatim, no model, no paraphrase.
        rule = self.overrides.match(question)
        if rule:
            answer = rule["answer"]
            if rule["source"]:
                answer += f'\n\nManba: {rule["source"]}'
            return dspy.Prediction(answer=answer, mode="override",
                                   citations=[{"code": "override", "code_title": rule["source"],
                                               "article": rule["id"], "lex_uz": ""}])

        refs = self.index.parse_references(question, context=context)
        resolved = [self.index.resolve(r) for r in refs]

        found = [rec for st, _, rec in resolved if st == "EXISTS"]
        notes = [
            self._missing(st, hint)
            for st, hint, _ in resolved
            if st in ("REPEALED", "OUT_OF_RANGE", "NO_CODE", "AMBIGUOUS_CODE")
        ]

        if refs:
            if found:
                body, ok = self._grounded(question, found)
                answer = ("\n".join(notes) + "\n\n" + body) if notes else body
                used, mode = found, "article-lookup"
            else:
                # every referenced article is missing -> answered without any LLM
                answer, used, mode, ok = "\n".join(notes), [], "deterministic", True
        else:
            keys = self.retriever.search(question)
            retrieved = [self.index.by_key[k] for k in keys if k in self.index.by_key]
            answer, ok = self._grounded(question, retrieved)
            # report only what the answer actually leans on, not every candidate
            used = self.index.cited_subset(answer, retrieved) if ok else []
            mode = "semantic"

        if used and ok:
            answer = f"{answer}\n\nManbalar:\n{_sources(used)}"

        return dspy.Prediction(
            answer=answer,
            mode=mode,
            citations=[
                {"code": a["code"], "code_title": a["code_title"],
                 "article": a["article_display"], "lex_uz": a["lex_uz_doc"]}
                for a in used
            ],
        )


_rag = None


def get_rag():
    global _rag
    if _rag is None:
        _rag = LegalRAG()
    return _rag
