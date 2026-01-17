from typing import Dict, Any, List

from schema_mapper_toolkit.matching.bm25_matcher import bm25_prepare, bm25_topk
from schema_mapper_toolkit.matching.dense_minilm_matcher import dense_topk
from schema_mapper_toolkit.matching.groq_reranker import groq_rerank
from schema_mapper_toolkit.matching.ensemble import manual_score, type_score, ensemble_score
from schema_mapper_toolkit.common.db_config import QdrantConfig


def build_column_text(col_id: str, col_type: str, descriptions: Dict[str, str]) -> str:
    desc = descriptions.get(col_id, "")
    return f"{col_id} type {col_type} description {desc}".strip()


def run_hybrid_matching_pipeline(
    source_cols: List[Dict[str, Any]],
    target_cols: List[Dict[str, Any]],
    source_descriptions: Dict[str, str],
    target_descriptions: Dict[str, str],
    qdrant_cfg: QdrantConfig,
    groq_api_key: str,
    groq_model: str,
    top_k: int = 10,
    final_top_k: int = 5,
    use_llm: bool = True,
):
    source_docs = []
    for c in source_cols:
        source_docs.append({
            "id": c["id"],
            "type": c["type"],
            "text": build_column_text(c["id"], c["type"], source_descriptions),
        })

    target_docs = []
    for c in target_cols:
        target_docs.append({
            "id": c["id"],
            "type": c["type"],
            "text": build_column_text(c["id"], c["type"], target_descriptions),
        })

    bm25 = bm25_prepare(target_docs)
    target_type_map = {t["id"]: t["type"] for t in target_docs}

    results = []

    for src in source_docs:
        query_text = src["text"]

        sparse = bm25_topk(bm25, target_docs, query_text, top_k)
        dense = dense_topk(qdrant_cfg, query_text, top_k)

        all_ids = set(sparse.keys()) | set(dense.keys())

        merged = []
        for cid in all_ids:
            bm25_s = sparse.get(cid, 0.0)
            dense_s = dense.get(cid, 0.0)

            man_s = manual_score(src["id"], cid)
            t_s = type_score(src["type"], target_type_map.get(cid, ""))

            final_s = ensemble_score(man_s, bm25_s, dense_s, t_s)

            merged.append({
                "candidate": cid,
                "manual_score": round(man_s, 4),
                "bm25_score": round(bm25_s, 4),
                "dense_score": round(dense_s, 4),
                "type_score": round(t_s, 4),
                "final_score": round(final_s, 4),
            })

        merged.sort(key=lambda x: x["final_score"], reverse=True)
        top_candidates = merged[:final_top_k]

        if not top_candidates:
            results.append({
                "source": src["id"],
                "source_type": src["type"],
                "best_match": None,
                "confidence": 0.0,
                "match_source": "none",
                "candidates": [],
            })
            continue

        if use_llm and groq_api_key:
            try:
                llm = groq_rerank(groq_api_key, groq_model, query_text, top_candidates)
                best_match = llm.get("best_match")
                confidence = float(llm.get("confidence", top_candidates[0]["final_score"]))

                cand_ids = {c["candidate"] for c in top_candidates}
                if best_match not in cand_ids:
                    best_match = top_candidates[0]["candidate"]
                    confidence = float(top_candidates[0]["final_score"])
                    match_source = "ensemble_fallback_invalid_llm"
                else:
                    match_source = "ensemble+llm"
            except Exception:
                best_match = top_candidates[0]["candidate"]
                confidence = float(top_candidates[0]["final_score"])
                match_source = "ensemble_fallback_llm_error"
        else:
            best_match = top_candidates[0]["candidate"]
            confidence = float(top_candidates[0]["final_score"])
            match_source = "ensemble"

        results.append({
            "source": src["id"],
            "source_type": src["type"],
            "best_match": best_match,
            "confidence": round(confidence, 4),
            "match_source": match_source,
            "candidates": top_candidates,
        })

    return {"match_count": len(results), "matches": results}
