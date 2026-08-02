"""Build both indexes. Run once, and again after any corpus update.

    python build_index.py            # reads ../data/articles/*.jsonl
    CORPUS_SOURCE=hf python build_index.py

Run this BEFORE starting vLLM so the embedding model has the GPU to itself
(~1 minute for 7,368 articles on an H100 / RTX PRO 6000).
"""
import glob
import json
import os
import sys

import config as C

FIELDS = ("code", "code_title", "article_id", "article_display", "article_raw",
          "kind", "title", "section", "chapter", "text", "lex_uz_doc")


def load_records():
    if C.CORPUS_SOURCE == "hf":
        from datasets import load_dataset
        ds = load_dataset(C.HF_DATASET, split=C.HF_SPLIT)
        rows = [dict(r) for r in ds]
    else:
        paths = sorted(glob.glob(os.path.join(C.LOCAL_ARTICLES, "*.jsonl")))
        if not paths:
            sys.exit(f"no .jsonl files under {C.LOCAL_ARTICLES}")
        rows = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return [{k: (r.get(k) or "") for k in FIELDS} for r in rows]


def context_doc(r):
    """Contextual retrieval: prepend each article's identity to its own text so
    an embedding carries 'which code / which article' as well as the wording.
    Anthropic's contextual-retrieval result, done statically -- no LLM pass,
    because this corpus already ships the exact context as metadata."""
    head = f'{r["code_title"]}, {r["article_display"]}'
    if r["title"]:
        head += f' — {r["title"]}'
    if r["chapter"]:
        head += f' ({r["chapter"]})'
    return (head + "\n" + r["text"])[: C.MAX_ARTICLE_CHARS]


def main():
    records = load_records()
    with open(C.INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    from corpus_index import CorpusIndex
    idx = CorpusIndex(records)
    print(f"rows={len(records)}  citable_articles={len(idx.articles)}  codes={len(idx.slugs)}")
    print(f"wrote {C.INDEX_JSON}")

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    try:
        client.delete_collection(C.COLLECTION)
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=C.EMBED_MODEL, device=C.BUILD_DEVICE
    )
    col = client.create_collection(
        C.COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

    arts = idx.articles  # front/end matter is never retrievable as an "article"
    B = 256
    for i in range(0, len(arts), B):
        chunk = arts[i : i + B]
        col.add(
            ids=[f'{r["code"]}::{r["article_id"]}' for r in chunk],
            documents=[context_doc(r) for r in chunk],
            metadatas=[{"code": r["code"], "article_id": r["article_id"]} for r in chunk],
        )
        print(f"  embedded {min(i + B, len(arts))}/{len(arts)}", flush=True)
    print("collection count:", col.count())


if __name__ == "__main__":
    main()
