"""Hybrid retrieval for natural-language questions.

Dense (bge-m3 over contextual embeddings) catches paraphrase; BM25 catches the
exact legal terms a dense model blurs ("neustoyka", "yuridik shaxs"). Reciprocal
rank fusion merges them. Returns keys only -- the pipeline joins them back to
full verbatim article text, so the model never sees a truncated article.
"""
import re

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

import config as C
from corpus_index import norm


def _tok(s):
    return re.findall(r"\w+", norm(s))


class Retriever:
    def __init__(self, articles):
        client = chromadb.PersistentClient(path=C.CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=C.EMBED_MODEL, device=C.QUERY_DEVICE
        )
        self.col = client.get_collection(C.COLLECTION, embedding_function=ef)
        self.keys = [(r["code"], r["article_id"]) for r in articles]
        self.bm25 = BM25Okapi(
            [_tok(f'{r["code_title"]} {r["article_display"]} {r["title"]} {r["text"]}')
             for r in articles]
        )

    def search(self, question, k=C.TOP_K):
        n = C.CANDIDATES
        res = self.col.query(query_texts=[question], n_results=n, include=["metadatas"])
        dense = [(m["code"], m["article_id"]) for m in res["metadatas"][0]]

        scores = self.bm25.get_scores(_tok(question))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        sparse = [self.keys[i] for i in top if scores[i] > 0]

        fused = {}
        for ranked in (dense, sparse):
            for rank, key in enumerate(ranked):
                fused[key] = fused.get(key, 0.0) + 1.0 / (C.RRF_K + rank)
        return sorted(fused, key=lambda key: -fused[key])[:k]
