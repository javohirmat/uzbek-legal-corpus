"""The pipeline: deterministic resolution -> grounded generation -> citation audit.

Order matters. Anything the deterministic layer can answer exactly (an article
number that exists, or provably does not) never reaches the model, so those
answers cannot be hallucinated. Only open questions go to generation, and what
comes back is audited against the articles that were actually supplied.
"""
import json
import re

import dspy

import config as C
from corpus_index import CorpusIndex, norm
from answer_rules import Overrides
from retriever import Retriever

_lm_kwargs = {}
if C.THINKING_MODE in ("true", "false"):
    # only override the template when explicitly forced; "auto" leaves the
    # decision to the model, which is what we want in production
    _lm_kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": C.THINKING_MODE == "true"}
    }

lm = dspy.LM(
    f"openai/{C.VLLM_MODEL}",
    api_base=C.VLLM_BASE,
    api_key=C.VLLM_KEY,
    temperature=C.TEMPERATURE,
    max_tokens=C.MAX_TOKENS,
    **_lm_kwargs,
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


CHAT_SYSTEM = (
    "Sen Tomaris — oʻzbek tili va madaniyati uchun yaratilgan sunʼiy intellekt "
    "yordamchisisan. Foydalanuvchiga oʻzbek tilida tabiiy, aniq va foydali javob ber."
)

# Does this message belong to the legal corpus at all? tomaris.ai is a general
# assistant: greetings, history questions and small talk must NOT be forced
# through statute retrieval, or they come back as
# "Berilgan moddalarda bunga javob yoʻq".
_LEGAL_HINT = re.compile(
    r"(modda|kodeks|qonun|huquq|jazo|sud|shartnoma|majburiyat|javobgarlik|"
    r"jinoyat|fuqarolik|mehnat|soliq|bojxona|ijara|nikoh|meros|farzandlikka|"
    r"davo|konstitutsiya|litsenziya|jarima|nafaqa|mulk|vorislik|ajrashish|"
    r"shartnomani|huquqiy|qonuniy|jinoiy|sudga|notarius)",
    re.IGNORECASE,
)


def _format(articles):
    return "\n\n".join(
        f'[{a["code_title"]} | {a["article_display"]}'
        + (f' | {a["title"]}' if a["title"] else "")
        + f']\n{a["text"][: C.MAX_ARTICLE_CHARS]}'
        for a in articles
    )


class UpstreamUnavailable(RuntimeError):
    """vLLM is down, restarting, or otherwise unreachable."""


def _as_text(x):
    """Coerce a model field to text.

    DSPy does not always hand back a plain string: depending on what the model
    emits, an output field can arrive as a dict or a list. Everything
    downstream (citation audit, normalisation) assumes text, so coerce once
    here rather than defending in each of those places.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # `reasoning_content` is included deliberately: with a reasoning parser
        # active and no closing </think> marker, vLLM files the entire answer
        # there and leaves content null. Losing that would discard a correct
        # answer we already paid to generate.
        for key in ("answer", "javob", "text", "content", "value", "reasoning_content"):
            if isinstance(x.get(key), str) and x[key].strip():
                return x[key]
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        return "\n".join(_as_text(i) for i in x)
    return str(x)


def _last_reasoning():
    """Pull reasoning_content off the most recent completion.

    DSPy's Prediction only carries the parsed output fields, so the model's
    thinking trace has to come from the LM's own history. Best-effort: an empty
    string simply means the UI shows no reasoning panel.

    Caveat: history is per-LM, not per-request, so under genuinely concurrent
    load this can attribute one request's trace to another. Fine for demo and
    light traffic; needs a per-call capture before heavy concurrent use.
    """
    try:
        resp = lm.history[-1].get("response")
        choices = getattr(resp, "choices", None) or resp.get("choices")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else choices[0].message
        rc = msg.get("reasoning_content") if isinstance(msg, dict) else getattr(msg, "reasoning_content", None)
        return rc if isinstance(rc, str) and rc.strip() else ""
    except Exception:
        return ""


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
        return _as_text(out[0] if isinstance(out, list) else out)

    def _grounded(self, question, articles):
        ctx = _format(articles)
        allowed = self.index.allowed_ids(articles)
        try:
            answer = _as_text(self.generate(question=question, articles=ctx).answer)
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
                answer = _as_text(self.generate(question=retry, articles=ctx).answer)
            except Exception as first:
                try:
                    answer = self._raw_call(retry, ctx)
                except Exception as second:
                    raise UpstreamUnavailable(str(second)) from first
            if self.index.bad_citations(answer, allowed):
                # We hold the verbatim articles the user asked about. Returning
                # "not enough information" while sitting on them is worse than
                # useless -- show the law itself instead of the model's take.
                verbatim = "\n\n".join(
                    f'{a["code_title"]}, {a["article_display"]}:\n{a["text"]}'
                    for a in articles
                )
                return (
                    "Quyida soʻralgan moddaning toʻliq matni keltirilgan:\n\n"
                    + verbatim,
                    False,
                    _last_reasoning(),
                )
        return answer, True, _last_reasoning()

    def _is_legal(self, question, refs):
        """Route to the corpus only when the message is actually about law."""
        if refs:                                        # "JK 173-modda"
            return True
        if self.index.find_codes(question):             # "Mehnat kodeksi ..."
            return True
        return bool(_LEGAL_HINT.search(norm(question)))

    def _chat(self, question, history):
        """General assistant turn: no retrieval, no citation audit, full
        conversation history so follow-ups make sense."""
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        messages += [m for m in history if m.get("content")][-8:]
        messages.append({"role": "user", "content": question})
        try:
            out = lm(messages=messages)
        except Exception as e:
            raise UpstreamUnavailable(str(e)) from e
        return _as_text(out[0] if isinstance(out, list) else out), _last_reasoning()

    # ---------------- entry point ----------------
    def forward(self, question, context="", history=()):
        # Customer-configured answers outrank everything, including the corpus:
        # if a bank has specified the reply to a question, that reply is the
        # answer -- verbatim, no model, no paraphrase.
        rule = self.overrides.match(question)
        if rule:
            answer = rule["answer"]
            if rule["source"]:
                answer += f'\n\nManba: {rule["source"]}'
            return dspy.Prediction(answer=answer, reasoning="", mode="override",
                                   citations=[{"code": "override", "code_title": rule["source"],
                                               "article": rule["id"], "lex_uz": ""}])

        refs = self.index.parse_references(question, context=context)

        # Route between "answer from the law" and "answer as an assistant".
        # An explicit signal decides immediately; otherwise let the corpus vote
        # by how close its nearest article is. Keyword lists silently miss
        # everyday phrasings ("ish haqi" is a Labour Code question that names
        # no code), and missing one means answering a legal question from the
        # model's memory -- the failure this system exists to prevent.
        explicit = self._is_legal(question, refs)
        retrieved, best = [], 1.0
        if not explicit:
            keys, best = self.retriever.search(question)
            if best > C.LEGAL_DISTANCE:
                answer, reasoning = self._chat(question, list(history))
                return dspy.Prediction(answer=answer, reasoning=reasoning,
                                       mode="chat", citations=[])
            retrieved = [self.index.by_key[k] for k in keys if k in self.index.by_key]

        resolved = [self.index.resolve(r) for r in refs]

        found = [rec for st, _, rec in resolved if st == "EXISTS"]
        notes = [
            self._missing(st, hint)
            for st, hint, _ in resolved
            if st in ("REPEALED", "OUT_OF_RANGE", "NO_CODE", "AMBIGUOUS_CODE")
        ]

        if refs:
            if found:
                body, ok, reasoning = self._grounded(question, found)
                answer = ("\n".join(notes) + "\n\n" + body) if notes else body
                used, mode = found, "article-lookup"
            else:
                # every referenced article is missing -> answered without any LLM
                answer, used, mode, ok = "\n".join(notes), [], "deterministic", True
                reasoning = ""
        else:
            if not retrieved:                       # explicit legal signal path
                keys, best = self.retriever.search(question)
                retrieved = [self.index.by_key[k] for k in keys if k in self.index.by_key]
            answer, ok, reasoning = self._grounded(question, retrieved)
            # report only what the answer actually leans on, not every candidate
            used = self.index.cited_subset(answer, retrieved) if ok else []
            mode = "semantic"

        if used and ok:
            answer = f"{answer}\n\nManbalar:\n{_sources(used)}"

        return dspy.Prediction(
            answer=answer,
            reasoning=reasoning,
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
