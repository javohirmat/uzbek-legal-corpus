"""OpenAI-compatible endpoint for the Vercel frontend.

Same contract the old rag_server exposed (POST /v1/chat/completions), so nothing
on the frontend changes except VAST_API_URL. `citations` is an extra top-level
field -- clients that ignore it are unaffected.
"""
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config as C
from pipeline import get_rag

app = FastAPI(title="Tomaris legal RAG")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def warm():
    idx = get_rag().index
    print(f"[ready] {len(idx.articles)} articles, {len(idx.slugs)} codes")


@app.get("/health")
def health():
    idx = get_rag().index
    return {"ok": True, "articles": len(idx.articles), "codes": len(idx.slugs)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    question = messages[-1]["content"] if messages else ""
    # earlier turns are used only to recover a code name for follow-ups
    # like "va 12-moddasi-chi?"
    context = "\n".join(
        m.get("content", "") for m in messages[-C.HISTORY_TURNS : -1]
    )

    print(f"\n[USER] {question}")
    result = get_rag()(question=question, context=context)
    print(f"[{result.mode}] {result.answer[:160]}...")

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": C.VLLM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "citations": result.citations,
            "retrieval_mode": result.mode,
        }
    )
