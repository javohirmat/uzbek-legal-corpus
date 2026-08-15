"""The pipeline: deterministic resolution -> grounded generation -> citation audit.

Order matters. Anything the deterministic layer can answer exactly (an article
number that exists, or provably does not) never reaches the model, so those
answers cannot be hallucinated. Only open questions go to generation, and what
comes back is audited against the articles that were actually supplied.
"""
import json
from datetime import datetime, timedelta, timezone

import dspy

import config as C
from corpus_index import CorpusIndex, norm
from answer_rules import Overrides
from query_expand import QueryExpander
import identity
from normalize_query import normalize_query
from retriever import Retriever
import situation_queries
from pack_context import pack_articles, format_grounding
from situation_prompt import (
    FORMAL_REGISTER_RU, FORMAL_REGISTER_UZ, SITUATION_SYSTEM_UZ,
    audit_fail_reply, generation_system, mostly_cyrillic, situation_system_for,
)
from legal_hints import LATIN_LEGAL_HINT, has_legal_cue

# Alias kept so older tests/grep that look for `_LEGAL_HINT` still find a gate.
_LEGAL_HINT = LATIN_LEGAL_HINT

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
    FORMAL_REGISTER_UZ
    + "Sen Oʻzbekiston qonunchiligi boʻyicha yordamchisan. Javobni FAQAT quyida "
    "berilgan moddalar matni asosida yoz. Har bir daʼvodan keyin (Kodeks nomi, "
    "N-modda) koʻrinishida iqtibos keltir. Agar berilgan moddalarda javob "
    "boʻlmasa, «Berilgan moddalarda bunga javob yoʻq» deb yoz. Berilmagan modda "
    "raqamini hech qachon oʻylab topma. Javobing qisqa va aniq boʻlsin — "
    "eng koʻpi 4-6 jumla. Modda matnini toʻliq koʻchirma, faqat savolga tegishli "
    "qismini tushuntir. Oxirgi qator alohida: Men yurist emasman. "
    "«qonun buzilgan» deb yozma."
)


class GroundedAnswer(dspy.Signature):
    """Rasmiy yozma o‘zbek; faqat berilgan moddalar; Aka yo‘q; yurist emassan."""

    articles = dspy.InputField(desc="tegishli qonun moddalari (toʻliq matn)")
    question = dspy.InputField(desc="foydalanuvchi vaziyati, moddalardan keyin")
    answer = dspy.OutputField(desc="iqtiboslangan javob (oʻzbek tilida)")


class SituationAnswer(dspy.Signature):
    """Vaziyat: faqat berilgan moddalar; oqibatlarni matndan iqtibos qil; yurist emassan."""

    articles = dspy.InputField(desc="tegishli qonun moddalari (toʻliq matn)")
    question = dspy.InputField(desc="foydalanuvchi vaziyati, moddalardan keyin")
    answer = dspy.OutputField(desc="iqtiboslangan javob (oʻzbek tilida)")


SituationAnswer.__doc__ = SITUATION_SYSTEM_UZ
GroundedAnswer.__doc__ = SYSTEM


_MONTHS_UZ = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul",
              "avgust", "sentabr", "oktabr", "noyabr", "dekabr"]


def chat_system(question=""):
    """Built per request so the date is never stale. A model with no clock
    answers "men bugungi sanani bilmayman", which reads as a broken assistant.

    Chat is not a lawyer: if a legal question still lands here, refuse to
    invent articles or punishments. Routing should have sent those to retrieve.
    """
    now = datetime.now(timezone.utc) + timedelta(hours=5)   # Asia/Tashkent
    today = f"{now.day}-{_MONTHS_UZ[now.month - 1]} {now.year}"
    if mostly_cyrillic(question):
        return (
            "Ты Tomaris — искусственный интеллект, созданный командой Tomaris AI "
            "в Узбекистане. Не называй другую компанию своим создателем. "
            + FORMAL_REGISTER_RU
            + "Ты не юрист. Если вопрос о законе, наказании, армии, аресте или "
            "штрафе — не угадывай статьи и меры наказания. Одним официальным "
            "предложением скажи, что нужен поиск по тексту закона. Не пиши "
            "«закон нарушен». "
            f"Сегодня: {today} (время Ташкента)."
        )
    return (
        "Sen Tomaris — oʻzbek tili va madaniyati uchun yaratilgan sunʼiy intellekt "
        "yordamchisisan. Seni Oʻzbekistondagi Tomaris AI jamoasi yaratgan; "
        "boshqa hech qanday loyiha yoki kompaniyani oʻz yaratuvching deb "
        "aytma. "
        + FORMAL_REGISTER_UZ
        + "Sen yurist emassan. Qonun, jazo, armiya, qamoq, jarima haqidagi "
        "savolda modda raqami yoki jazoni taxmin qilma. Bitta rasmiy jumla: "
        "bunday savol qonun matnini qidirishni talab qiladi. «qonun buzilgan» "
        "va «Baraka toping» deb yozma. "
        f"Bugungi sana: {today} (Toshkent vaqti). "
        "Foydalanuvchi salomlashsa, rasmiy salom bilan javob ber — uning "
        "soʻzini takrorlama."
    )

# Does this message belong to the legal corpus at all? tomaris.ai is a general
# assistant: greetings, history questions and small talk must NOT be forced
# through statute retrieval, or they come back as
# "Berilgan moddalarda bunga javob yoʻq".
def _format(articles):
    packed = pack_articles(articles)
    return "\n\n".join(
        f'[{a["code_title"]} | {a["article_display"]}'
        + (f' | {a["title"]}' if a.get("title") else "")
        + f']\n{a["text"][: C.MAX_ARTICLE_CHARS]}'
        for a in packed
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
        self.expander = QueryExpander.load(C.SYNONYMS_JSON)
        self.generate = dspy.Predict(GroundedAnswer)
        self.generate_situation = dspy.Predict(SituationAnswer)

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
    def _raw_call(self, question, articles, system=SYSTEM):
        """Fallback when DSPy cannot parse the structured reply (fine-tuned
        models do not always honour field markers)."""
        out = lm(messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": format_grounding(articles, question)},
        ])
        return _as_text(out[0] if isinstance(out, list) else out)

    def _grounded(self, question, articles, generate=None, system=None,
                  skip_dspy=False):
        if generate is None:
            generate = self.generate
        if system is None:
            system = SYSTEM
        ctx = _format(articles)
        allowed = self.index.allowed_ids(articles)
        try:
            if skip_dspy:
                answer = self._raw_call(question, articles, system=system)
            else:
                answer = _as_text(generate(question=question, articles=ctx).answer)
        except Exception as first:
            # A structured-output parse failure is recoverable via a plain call;
            # an unreachable vLLM is not, and must not surface as a raw traceback.
            try:
                answer = self._raw_call(question, articles, system=system)
            except Exception as second:
                raise UpstreamUnavailable(str(second)) from first

        bad = self.index.bad_citations(answer, allowed)
        if bad:
            retry = (
                f"{question}\n\n[DIQQAT: faqat berilgan moddalarga iqtibos qil. "
                f'Quyidagilar berilmagan: {", ".join(sorted(set(bad)))}]'
            )
            try:
                if skip_dspy:
                    answer = self._raw_call(retry, articles, system=system)
                else:
                    answer = _as_text(generate(question=retry, articles=ctx).answer)
            except Exception as first:
                try:
                    answer = self._raw_call(retry, articles, system=system)
                except Exception as second:
                    raise UpstreamUnavailable(str(second)) from first
            if self.index.bad_citations(answer, allowed):
                # Do not dump the supplied statutes: six articles is a 10k
                # Uzbek wall, and a Russian question must not get that dump.
                return audit_fail_reply(question, articles), False, _last_reasoning()
        return answer, True, _last_reasoning()

    def _refs(self, question, context=""):
        """Parse N-modda on the raw string first so SMS maps cannot eat 999.

        Cyrillic «модда» only reaches the parser after query normalization.
        """
        refs = self.index.parse_references(question, context=context)
        if refs:
            return refs
        normalized = normalize_query(question)
        if normalized != question:
            return self.index.parse_references(normalized, context=context)
        return refs

    def _is_legal(self, question, refs):
        """Route to the corpus only when the message is actually about law."""
        if refs:                                        # "JK 173-modda"
            return True
        for q in (question, normalize_query(question)):
            if self.index.find_codes(q):                # "Mehnat kodeksi ..."
                return True
            if has_legal_cue(q) or _LEGAL_HINT.search(norm(q)):
                return True
        return False

    def _lookup(self, keys):
        return [self.index.by_key[k] for k in keys if k in self.index.by_key]

    def _rewrite(self, question):
        """One cheap 27B call, thinking off. Empty string on timeout/error."""
        try:
            from openai import OpenAI
            client = OpenAI(base_url=C.VLLM_BASE, api_key=C.VLLM_KEY,
                            timeout=C.REWRITE_TIMEOUT)
            resp = client.chat.completions.create(
                model=C.VLLM_MODEL,
                messages=[
                    {"role": "system", "content": situation_queries.REWRITE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=C.REWRITE_MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = resp.choices[0].message
            return (getattr(msg, "content", None) or "") if msg else ""
        except Exception as e:
            print(f"[rewrite-fallback] {type(e).__name__}: {e}")
            return ""

    def _situation_retrieve(self, question):
        """Multi-query RRF + per-code cap. Used only when no N-modda was parsed."""
        queries = situation_queries.queries_for(
            question, self.expander, complete_fn=self._rewrite
        )
        print(f"[situation-queries] {queries}")
        keys, best = self.retriever.search_multi(queries)
        return self._lookup(keys), best

    def _chat(self, question, history):
        """General assistant turn: no retrieval, no citation audit, full
        conversation history so follow-ups make sense."""
        messages = [{"role": "system", "content": chat_system(question)}]
        messages += [m for m in history if m.get("content")][-8:]
        messages.append({"role": "user", "content": question})
        try:
            out = lm(messages=messages, temperature=C.CHAT_TEMPERATURE)
        except Exception as e:
            raise UpstreamUnavailable(str(e)) from e
        return _as_text(out[0] if isinstance(out, list) else out), _last_reasoning()

    # ---------------- entry point ----------------
    def forward(self, question, context="", history=()):
        # Customer-configured answers outrank everything, including the corpus:
        # if a bank has specified the reply to a question, that reply is the
        # answer -- verbatim, no model, no paraphrase.
        # Who/what am I, and prompt-injection attempts. Answered from a fixed
        # string without invoking the model, so there is nothing to talk out of
        # its instructions.
        fixed = identity.match(question)
        if fixed:
            return dspy.Prediction(answer=fixed, reasoning="", mode="identity",
                                   citations=[])

        rule = self.overrides.match(question)
        if rule:
            answer = rule["answer"]
            if rule["source"]:
                answer += f'\n\nManba: {rule["source"]}'
            return dspy.Prediction(answer=answer, reasoning="", mode="override",
                                   citations=[{"code": "override", "code_title": rule["source"],
                                               "article": rule["id"], "lex_uz": ""}])

        refs = self._refs(question, context=context)

        # Route between "answer from the law" and "answer as an assistant".
        # An explicit signal decides immediately; otherwise let the corpus vote
        # by how close its nearest article is. Keyword lists silently miss
        # everyday phrasings ("ish haqi" is a Labour Code question that names
        # no code), and missing one means answering a legal question from the
        # model's memory -- the failure this system exists to prevent.
        explicit = self._is_legal(question, refs)
        retrieved, best = [], 1.0
        if not explicit:
            keys, best = self.retriever.search(self.expander.expand(question))
            if best > C.LEGAL_DISTANCE:
                answer, reasoning = self._chat(question, list(history))
                return dspy.Prediction(answer=answer, reasoning=reasoning,
                                       mode="chat", citations=[])
            retrieved = self._lookup(keys)

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
            # Story / open legal question: rewrite into several searches so one
            # keyword-heavy code cannot fill the whole window.
            retrieved, best = self._situation_retrieve(question)
            sit_sys = situation_system_for(question)
            answer, ok, reasoning = self._grounded(
                question, retrieved,
                generate=self.generate_situation, system=sit_sys,
                skip_dspy=sit_sys is not SITUATION_SYSTEM_UZ,
            )
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


# ---------------------------------------------------------------- streaming
# The non-streaming path above buffers the whole answer before sending it, so
# the user stares at nothing for the full generation. Streaming emits tokens as
# the model writes them -- but the citation audit needs complete text, so it is
# applied incrementally: text is released only up to the last whitespace before
# a holdback window, which guarantees no "N-modda" is ever split across the
# boundary, and every citation is checked before the user can see it.
from openai import OpenAI  # noqa: E402

_client = OpenAI(base_url=C.VLLM_BASE, api_key=C.VLLM_KEY)
HOLDBACK = 32          # chars; longest citation form is well under this


class BadCitation(RuntimeError):
    """Model cited an article it was not given; stop the stream."""


def _stream_lm(messages, temperature=C.TEMPERATURE):
    """Yield ("reasoning"|"content", text) as vLLM produces them."""
    stream = _client.chat.completions.create(
        model=C.VLLM_MODEL, messages=messages, stream=True,
        temperature=temperature, max_tokens=C.MAX_TOKENS,
    )
    for event in stream:
        if not event.choices:
            continue
        d = event.choices[0].delta
        rc = getattr(d, "reasoning_content", None)
        if rc:
            yield "reasoning", rc
        if d.content:
            yield "content", d.content


def _safe_cut(buf):
    """Largest index we can release without splitting a word (so never a citation)."""
    limit = len(buf) - HOLDBACK
    if limit <= 0:
        return 0
    cut = buf.rfind(" ", 0, limit)
    return cut + 1 if cut > 0 else 0


def stream_answer(rag, question, context="", history=()):
    """Yield ("reasoning"|"content"|"done", payload) for a streaming request.

    Fast paths (identity/override/deterministic) emit their fixed text at once;
    they never call the model. Everything else streams token by token.
    """
    fixed = identity.match(question)
    if fixed:
        yield "content", fixed
        yield "done", {"mode": "identity", "citations": [], "answer": fixed}
        return

    rule = rag.overrides.match(question)
    if rule:
        text = rule["answer"] + (f'\n\nManba: {rule["source"]}' if rule["source"] else "")
        yield "content", text
        yield "done", {"mode": "override", "answer": text,
                       "citations": [{"code": "override", "code_title": rule["source"],
                                      "article": rule["id"], "lex_uz": ""}]}
        return

    refs = rag._refs(question, context=context)
    explicit = rag._is_legal(question, refs)
    retrieved = []
    if not explicit:
        keys, best = rag.retriever.search(rag.expander.expand(question))
        if best > C.LEGAL_DISTANCE:                      # ordinary conversation
            msgs = [{"role": "system", "content": chat_system(question)}]
            msgs += [m for m in history if m.get("content")][-8:]
            msgs.append({"role": "user", "content": question})
            yield from _passthrough(msgs, mode="chat")
            return
        retrieved = rag._lookup(keys)

    resolved = [rag.index.resolve(r) for r in refs]
    found = [rec for st, _, rec in resolved if st == "EXISTS"]
    notes = [rag._missing(st, h) for st, h, _ in resolved
             if st in ("REPEALED", "OUT_OF_RANGE", "NO_CODE", "AMBIGUOUS_CODE")]

    if refs and not found:                               # deterministic refusal
        text = "\n".join(notes)
        yield "content", text
        yield "done", {"mode": "deterministic", "citations": [], "answer": text}
        return

    if found:
        articles, mode = found, "article-lookup"
    else:
        articles, _ = rag._situation_retrieve(question)
        mode = "semantic"

    if notes:
        yield "content", "\n".join(notes) + "\n\n"

    allowed = rag.index.allowed_ids(articles)
    sys_prompt = generation_system(refs, SYSTEM, question)
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": format_grounding(articles, question)}]

    buf, emitted = "", []
    try:
        for kind, piece in _stream_lm(msgs):
            if kind == "reasoning":
                yield "reasoning", piece
                continue
            buf += piece
            cut = _safe_cut(buf)
            if cut:
                chunk, buf = buf[:cut], buf[cut:]
                if rag.index.bad_citations(chunk, allowed):
                    raise BadCitation
                emitted.append(chunk)
                yield "content", chunk
        if rag.index.bad_citations(buf, allowed):
            raise BadCitation
        emitted.append(buf)
        yield "content", buf
    except BadCitation:
        text = "\n\n" + audit_fail_reply(question, articles)
        yield "content", text
        emitted.append(text)
    except Exception as e:
        raise UpstreamUnavailable(str(e)) from e

    answer = "".join(emitted)
    cited = rag.index.cited_subset(answer, articles) if mode == "semantic" else articles
    if cited:
        src = "\n\nManbalar:\n" + _sources(cited)
        yield "content", src
        answer += src
    yield "done", {"mode": mode, "answer": answer,
                   "citations": [{"code": a["code"], "code_title": a["code_title"],
                                  "article": a["article_display"],
                                  "lex_uz": a["lex_uz_doc"]} for a in cited]}


def _passthrough(messages, mode):
    """Stream a plain assistant turn -- no retrieval, nothing to audit."""
    parts = []
    try:
        for kind, piece in _stream_lm(messages, temperature=C.CHAT_TEMPERATURE):
            if kind == "reasoning":
                yield "reasoning", piece
            else:
                parts.append(piece)
                yield "content", piece
    except Exception as e:
        raise UpstreamUnavailable(str(e)) from e
    yield "done", {"mode": mode, "answer": "".join(parts), "citations": []}
