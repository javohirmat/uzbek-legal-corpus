"""Hybrid retrieval for natural-language questions.

Dense (bge-m3 over contextual embeddings) catches paraphrase; BM25 catches the
exact legal terms a dense model blurs ("neustoyka", "yuridik shaxs"). Reciprocal
rank fusion merges them. Returns keys only -- the pipeline joins them back to
full verbatim article text, so the model never sees a truncated article.

Situational questions search several queries and RRF-fuse the lists, then cap
how many articles any one code may occupy in the final window so a keyword-
heavy neighbour (Suv 72) cannot crowd out the governing family (FK ijara).
"""
import re

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

import config as C
from corpus_index import norm
from code_keywords import boost_matched_codes
from situation_queries import cap_per_code, exclude_unmentioned_soliq, rrf_fuse


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
        self.titles = {
            (r["code"], r["article_id"]): r.get("title") or "" for r in articles
        }
        self.bm25 = BM25Okapi(
            [_tok(f'{r["code_title"]} {r["article_display"]} {r["title"]} {r["text"]}')
             for r in articles]
        )

    def search(self, question, k=C.TOP_K):
        """Returns (keys, best_distance). The distance lets the caller judge
        whether the corpus actually has anything to say about this question --
        keyword lists cannot cover a whole language."""
        n = C.CANDIDATES
        res = self.col.query(query_texts=[question], n_results=n,
                             include=["metadatas", "distances"])
        dense = [(m["code"], m["article_id"]) for m in res["metadatas"][0]]
        dists = res.get("distances") or [[]]
        best = min(dists[0]) if dists[0] else 1.0

        scores = self.bm25.get_scores(_tok(question))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        sparse = [self.keys[i] for i in top if scores[i] > 0]

        fused = rrf_fuse([dense, sparse])
        return fused[:k], best

    def _title_prefer(self, keys, questions):
        """Keys whose article title appears in the search queries.

        Used so the per-code cap cannot drop JK 169 (Oʻgʻrilik) when 166/164
        crowd Jinoyat. Does not invent article numbers — titles come from the
        index, phrases from the queries already built.
        """
        blob = norm(" ".join(str(q) for q in questions if q))
        if not blob:
            return []
        prefer = []
        for key in keys:
            title = self.titles.get(key) or ""
            folded = norm(title)
            if folded and folded in blob:
                prefer.append(key)
        return prefer

    def search_multi(self, questions, k=C.TOP_K, cap=None, boost_codes=None):
        """Search each query for CANDIDATES, RRF-fuse the lists, cap per code.

        `boost_codes` promotes already-retrieved keys from matched codes. It
        never invents articles and never becomes the only retrieval path.
        """
        cap = C.PER_CODE_CAP if cap is None else cap
        lists, best = [], 1.0
        for q in questions:
            if not q or not str(q).strip():
                continue
            keys, dist = self.search(q, k=C.CANDIDATES)
            lists.append(keys)
            if dist < best:
                best = dist
        if not lists:
            return [], best
        fused = exclude_unmentioned_soliq(rrf_fuse(lists), questions)
        fused = boost_matched_codes(fused, boost_codes)
        prefer = self._title_prefer(fused, questions)
        return cap_per_code(fused, cap=cap, limit=k, prefer=prefer), best
