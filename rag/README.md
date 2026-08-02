# Tomaris legal RAG

Retrieval + anti-hallucination layer in front of `tomaris/Tomaris.ai` + the DPO
LoRA, served by vLLM. Answers questions about the 25 audited Uzbek legal codes
in this repo (7,368 citable articles).

## Answer paths

Every request takes exactly one of five paths, reported back as `retrieval_mode`
(measured on an RTX PRO 6000 Blackwell, 27B + LoRA, ~23 tok/s):

| mode | what it means | model | measured |
|---|---|---|---|
| `override` | a customer-configured answer in `overrides.json` | no | 0.007 s |
| `deterministic` | the article does not exist (repealed / out of range / ambiguous) | no | 0.005 s |
| `chat` | not a legal question — answered as a normal assistant | yes | 4–18 s |
| `article-lookup` | a specific article was named and found; answered from its verbatim text | yes | ~16 s |
| `semantic` | open legal question; hybrid retrieval then grounded generation | yes | 25–34 s |

The first two never reach the model, so they keep working even while vLLM is
restarting, and they cannot be hallucinated by construction.

### Legal or not?

This serves a general assistant, so most messages are not about law. Routing:

1. an article reference, a code name, or legal vocabulary → the corpus;
2. otherwise retrieval runs first and the nearest article's cosine distance
   decides (`LEGAL_DISTANCE`, default 0.45).

Step 2 exists because a keyword list cannot cover a language: *"Ish haqi qancha
muddatda toʻlanishi kerak?"* is a Labour Code question naming neither a code nor
any legal term, and routing it to chat would answer a legal question from the
model's memory — the exact failure this system exists to prevent. Retrieval
catches it; cooking and history questions stay in chat.

## Customer-configured answers

`overrides.json` lets a customer fix the reply to questions they care about —
what a bank needs before putting this in front of its own users:

```json
[{ "any":    ["kredit foizi", "foiz stavkasi"],
   "not":    ["ipoteka"],
   "answer": "Isteʼmol krediti boʻyicha yillik stavka — 24%.",
   "source": "Bank tariflari (2026-08)" }]
```

Matched before retrieval and before generation, so the configured answer is
returned verbatim and cannot be reworded. Matching ignores okina/apostrophe
variants, so `isteʼmol`, `iste'mol` and `istemol` all hit the same rule. The
`not` conditions keep rules from swallowing neighbouring questions: with the
rule above, *"Ipoteka kredit foizi"* falls through to normal RAG. Edit the file
and `supervisorctl restart tomaris-rag` to apply.

## Why it cannot invent articles

Three layers, only one of which involves the model:

1. **Deterministic resolution (no LLM).** Every `N-modda` in the question is
   resolved against an exact map of all 7,368 article IDs *before* generation.
   Exists → the verbatim article is fetched. Doesn't exist → a templated answer
   that distinguishes *repealed* (inside the code's range) from *out of range*
   (past its last article), with zero model involvement. `JK 999-modda` can
   never produce prose.
2. **Grounded generation.** The model only ever sees full retrieved articles and
   is instructed to cite `(Kodeks, N-modda)`.
3. **Citation audit.** Every `N-modda` in the generated answer is checked
   against the articles actually supplied *plus the articles those texts
   cross-reference* (21% of this corpus cites other articles, so ignoring that
   would reject faithful quotation). Unknown citation → one retry with explicit
   feedback → refusal. An invented number cannot reach the user.

Ambiguity is asked about, never guessed: `SK` matches four codes
(soliq/saylov/suv/shaharsozlik), `BK` matches two, and a bare `11-modda` with no
code named returns a clarify question listing the documents that contain it.

Run the guardrails against the real corpus at any time — no GPU, no network:

```bash
python test_guardrails.py
```

## Renting the box

Two hard requirements, both learned the expensive way:

- **CUDA driver ≥ 12.8.** A 12.6-driver box makes current vLLM/PyTorch wheels
  fail with `The NVIDIA driver on your system is too old`, and every attempted
  downgrade cascades into torch/torchaudio/transformers ABI breakage.
- **Disk ≥ 150 GB** (54 GB model + adapter + embeddings + wheels).

## Setup

**Two virtualenvs, deliberately.** vLLM pins torch and transformers hard;
`sentence-transformers` will upgrade them out from under it. The two processes
share nothing but a port number, so keeping their dependencies apart removes the
entire class of failure that ate the last deploy.

```bash
# corpus + code travel together
git clone https://github.com/javohirmat/uzbek-legal-corpus.git /workspace/repo

# venv 1 - inference
python3 -m venv /workspace/venv-vllm
/workspace/venv-vllm/bin/pip install vllm huggingface_hub

# venv 2 - retrieval
python3 -m venv /workspace/venv-rag
/workspace/venv-rag/bin/pip install -r /workspace/repo/rag/requirements.txt

# weights
/workspace/venv-vllm/bin/hf auth login
cd /workspace
HF_HUB_DISABLE_XET=1 /workspace/venv-vllm/bin/hf download tomaris/Tomaris.ai --local-dir /workspace/model
/workspace/venv-vllm/bin/hf download tomaris/Tomaris-dpo-v1 --local-dir /workspace/dpo_adapter
```

## Build the index

Do this **before** starting vLLM, while the GPU is still free (~1 min).

```bash
cd /workspace/repo/rag
/workspace/venv-rag/bin/python build_index.py
```

Reads `../data/articles/*.jsonl` directly — no Hugging Face round trip, so the
index always matches the audited corpus in this commit. Set `CORPUS_SOURCE=hf`
to pull from the Hub instead.

## Run

Both processes run as **supervisor services**, not `tmux` or `nohup` — they then
restart on crash and their logs reach the Vast portal. The two wrapper scripts
and configs are created by `install_services.sh`:

```bash
bash /workspace/repo/rag/install_services.sh
supervisorctl status | grep tomaris
```

`--language-model-only` is required in the vLLM service: the base is a Qwen3-VL
checkpoint and without it vLLM tries to load an image processor and dies with
`Can't load image processor`.

**vLLM binds `127.0.0.1` and deliberately gets no `portal.yaml` entry.** It has
no authentication of its own, so exposing it would let anyone with the URL spend
your GPU. Only the RAG server is public, and only behind the Caddy token edge.

Restart after a code change:

```bash
cd /workspace/repo && git pull && supervisorctl restart tomaris-rag
```

Never `pkill -f uvicorn…`: the pattern matches the very shell running it and
kills your own session mid-command. Use `supervisorctl`, or kill by PID.

## Verify

```bash
curl -s localhost:8000/health

# exists -> verbatim article + summary + sources
curl -s -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Konstitutsiyaning 149-moddasida nima deyilgan?"}]}'

# does not exist -> refusal, no LLM call at all
curl -s -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"JK 999-modda nima deydi?"}]}'
```

The second must answer `Jinoyat kodeksida 999-modda yoʻq. Oxirgi modda —
302-modda.` and report `"retrieval_mode": "deterministic"`.

## Frontend

The endpoint is OpenAI-compatible and unchanged from the old server, so only
`VAST_API_URL` moves — point it at the Vast public mapping for port 8000.
Responses carry two extra top-level fields that clients may ignore:
`citations` (code, article, lex.uz doc id) and `retrieval_mode`
(`article-lookup` | `deterministic` | `semantic`).

## Latency knob

Query-time embeddings default to CPU so they never contend with vLLM for VRAM
(~200–400 ms, negligible beside 27B generation). If you drop vLLM to
`--gpu-memory-utilization 0.85`, set `QUERY_DEVICE=cuda` for ~15 ms instead.
