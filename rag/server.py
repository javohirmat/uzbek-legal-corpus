"""OpenAI-compatible endpoint for the Vercel frontend.

Same contract the old rag_server exposed (POST /v1/chat/completions), so nothing
on the frontend changes except VAST_API_URL. `citations` is an extra top-level
field -- clients that ignore it are unaffected.
"""
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import config as C
from pipeline import UpstreamUnavailable, get_rag, stream_answer

app = FastAPI(title="Tomaris legal RAG")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def tolerate_doubled_base(request: Request, call_next):
    """Accept a base URL that already includes the endpoint path.

    Clients configured with VAST_API_URL=".../v1/chat/completions" append their
    own "/v1/..." and arrive as "/v1/chat/completions/v1/models". Rather than
    404 and look like an outage, strip the duplicated prefix.
    """
    path = request.url.path
    marker = "/v1/chat/completions/v1/"
    if marker in path:
        request.scope["path"] = path[path.index(marker) + len("/v1/chat/completions"):]
    return await call_next(request)


@app.on_event("startup")
def warm():
    idx = get_rag().index
    print(f"[ready] {len(idx.articles)} articles, {len(idx.slugs)} codes")


@app.get("/health")
def health():
    idx = get_rag().index
    return {"ok": True, "articles": len(idx.articles), "codes": len(idx.slugs)}


@app.get("/v1/models")
def models():
    """OpenAI-compatible model list.

    Clients probe this to decide whether a backend is reachable -- without it
    the frontend reports "not connected" and falls back to demo responses even
    though chat completions work.
    """
    return {
        "object": "list",
        "data": [{
            "id": C.VLLM_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tomaris",
        }],
    }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    question = messages[-1]["content"] if messages else ""
    history = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
    # earlier turns also let a follow-up like "va 12-moddasi-chi?" inherit its code
    context = "\n".join(
        m.get("content", "") for m in messages[-C.HISTORY_TURNS : -1]
    )
    want_stream = bool(body.get("stream"))

    t0 = time.time()
    print(f"\n[USER] {question}{'  (stream)' if want_stream else ''}")

    def run():
        result = get_rag()(question=question, context=context, history=history)
        print(f"[{result.mode}] {result.answer[:160]}...")
        try:
            with open(C.CHAT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "question": question,
                    "mode": result.mode,
                    "answer": result.answer,
                    "citations": result.citations,
                    "seconds": round(time.time() - t0, 1),
                }, ensure_ascii=False) + "\n")
        except Exception as e:                  # logging must never break a reply
            print(f"[chat-log failed] {e}")
        return result

    if want_stream:
        # Generation happens INSIDE the generator, and the first chunk goes out
        # before it starts. Building the answer first made time-to-first-byte
        # equal to full generation (~30s on the semantic path), so clients hit
        # their own timeout and showed "couldn't reach the model" while the
        # server was happily producing a correct answer.
        def sse():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            base = {"id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": C.VLLM_MODEL}

            def frame(delta, finish=None, extra=None):
                d = {**base, "choices": [{"index": 0, "delta": delta,
                                          "finish_reason": finish}]}
                if extra:
                    d.update(extra)
                return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

            yield frame({"role": "assistant"})      # flushes immediately

            done, reasoning_parts = None, []
            try:
                for kind, payload in stream_answer(
                        get_rag(), question, context=context, history=history):
                    if kind == "done":
                        done = payload
                    elif kind == "reasoning":
                        reasoning_parts.append(payload)
                        yield frame({"reasoning_content": payload})
                    else:
                        yield frame({"content": payload})
            except UpstreamUnavailable as e:
                print(f"[upstream-unavailable] {e}")
                yield frame({"content": "Til modeli hozircha ishga tushmoqda. "
                                        "Bir necha daqiqadan soʻng qayta urinib koʻring."})
                yield frame({}, "stop")
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                print(f"[stream-failed] {type(e).__name__}: {e}")
                yield frame({"content": "Javob tayyorlashda xatolik yuz berdi. "
                                        "Qayta urinib koʻring."})
                yield frame({}, "stop")
                yield "data: [DONE]\n\n"
                return

            done = done or {"mode": "unknown", "answer": "", "citations": []}
            print(f"[{done['mode']}] {done['answer'][:160]}...")
            try:
                with open(C.CHAT_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "question": question, "mode": done["mode"],
                        "answer": done["answer"], "citations": done["citations"],
                        "seconds": round(time.time() - t0, 1),
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[chat-log failed] {e}")

            yield frame({}, "stop", {"citations": done["citations"],
                                     "retrieval_mode": done["mode"]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no",
                                          "Connection": "keep-alive"})

    try:
        # run() blocks for the whole generation (up to a minute). Straight on
        # the event loop that freezes every other request -- /health, streams,
        # even the instant deterministic answers -- until it finishes.
        result = await run_in_threadpool(run)
    except UpstreamUnavailable as e:
        print(f"[upstream-unavailable] {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {"type": "upstream_unavailable", "message": str(e)},
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Til modeli hozircha ishga tushmoqda. "
                                   "Bir necha daqiqadan soʻng qayta urinib koʻring.",
                    },
                    "finish_reason": "stop",
                }],
            },
        )

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": C.VLLM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.answer,
                        # the frontend's collapsible "Reasoning" panel reads this;
                        # empty on the override/deterministic paths, which never
                        # invoke the model
                        "reasoning_content": result.reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "citations": result.citations,
            "retrieval_mode": result.mode,
        }
    )
