from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi


def bm25_prepare(target_docs: List[Dict[str, Any]]) -> BM25Okapi:
    corpus = [d["text"].lower().split() for d in target_docs]
    return BM25Okapi(corpus)


def bm25_topk(bm25: BM25Okapi, target_docs: List[Dict[str, Any]], query: str, top_k: int):
    scores = bm25.get_scores(query.lower().split())
    scores = np.array(scores, dtype=float)

    if scores.max() > 0:
        scores = scores / scores.max()

    idx = np.argsort(scores)[::-1][:top_k]
    return {target_docs[i]["id"]: float(scores[i]) for i in idx}
