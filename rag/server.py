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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import config as C
from pipeline import UpstreamUnavailable, get_rag

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
    try:
        result = get_rag()(question=question, context=context, history=history)
    except UpstreamUnavailable as e:
        # vLLM restarting or down. Deterministic answers still work (they never
        # call it), so only generation-backed queries land here.
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
    print(f"[{result.mode}] {result.answer[:160]}...")

    # Durable transcript. The supervisor log restarts with the process, so the
    # only record of what users actually asked was being lost on every deploy.
    # This file also becomes training data: real questions, real phrasings.
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
    except Exception as e:                      # logging must never break a reply
        print(f"[chat-log failed] {e}")

    if want_stream:
        # The citation audit needs the finished answer, so generation cannot be
        # streamed through. Emit the completed answer as OpenAI-style chunks so
        # streaming clients render it instead of waiting for an event that
        # never arrives.
        def sse():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            base = {"id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": C.VLLM_MODEL}
            first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                          "finish_reason": None}]}
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            text = result.answer
            for i in range(0, len(text), 48):
                chunk = {**base, "choices": [{"index": 0,
                                              "delta": {"content": text[i:i + 48]},
                                              "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            last = {**base, "citations": result.citations,
                    "retrieval_mode": result.mode,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

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
