from typing import Dict, Any, List, Tuple, Optional

from schema_matching_toolkit.sparse_bm25 import bm25_match
from schema_matching_toolkit.minilm_dense_matcher import match_source_to_target_dense
from schema_matching_toolkit.mpnet_embedding_matcher import mpnet_dense_match
from schema_matching_toolkit.common.db_config import QdrantConfig


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_scores(candidate_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize dict scores to 0..1 range using max normalization.
    """
    if not candidate_scores:
        return {}

    max_score = max(candidate_scores.values()) if candidate_scores else 0.0
    if max_score <= 0:
        return {k: 0.0 for k in candidate_scores.keys()}

    return {k: v / max_score for k, v in candidate_scores.items()}


def _collect_candidates(
    bm25_res: Dict[str, Any],
    minilm_res: Dict[str, Any],
    mpnet_res: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Output:
    {
      "source_col": {
         "target_col": {
            "bm25": score,
            "minilm": score,
            "mpnet": score
         }
      }
    }
    """
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ---- BM25 (Top-1)
    for row in bm25_res.get("matches", []):
        src = row.get("source")
        tgt = row.get("best_match")
        score = _safe_float(row.get("score", 0.0))

        if not src or not tgt:
            continue

        combined.setdefault(src, {}).setdefault(tgt, {})
        combined[src][tgt]["bm25"] = score

    # ---- MiniLM (top_k candidates)
    for row in minilm_res.get("matches", []):
        src = row.get("source")
        if not src:
            continue

        for cand in row.get("candidates", []):
            tgt = cand.get("target")
            score = _safe_float(cand.get("score", 0.0))
            if not tgt:
                continue

            combined.setdefault(src, {}).setdefault(tgt, {})
            combined[src][tgt]["minilm"] = score

    # ---- MPNet (top_k candidates)
    for row in mpnet_res.get("matches", []):
        src = row.get("source")
        if not src:
            continue

        for cand in row.get("candidates", []):
            tgt = cand.get("target")
            score = _safe_float(cand.get("score", 0.0))
            if not tgt:
                continue

            combined.setdefault(src, {}).setdefault(tgt, {})
            combined[src][tgt]["mpnet"] = score

    return combined


def _pick_best_candidate(
    candidate_map: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
    """
    candidate_map:
      {
        "target_col": {"bm25":0.2, "minilm":0.9, "mpnet":0.8}
      }
    """
    # Normalize each method scores separately for fairness
    bm25_scores = {t: v.get("bm25", 0.0) for t, v in candidate_map.items()}
    minilm_scores = {t: v.get("minilm", 0.0) for t, v in candidate_map.items()}
    mpnet_scores = {t: v.get("mpnet", 0.0) for t, v in candidate_map.items()}

    bm25_norm = _normalize_scores(bm25_scores)
    minilm_norm = _normalize_scores(minilm_scores)
    mpnet_norm = _normalize_scores(mpnet_scores)

    ranked = []

    for tgt in candidate_map.keys():
        s_bm25 = bm25_norm.get(tgt, 0.0)
        s_minilm = minilm_norm.get(tgt, 0.0)
        s_mpnet = mpnet_norm.get(tgt, 0.0)

        final_score = (
            weights.get("bm25", 0.0) * s_bm25
            + weights.get("minilm", 0.0) * s_minilm
            + weights.get("mpnet", 0.0) * s_mpnet
        )

        ranked.append(
            {
                "candidate": tgt,
                "bm25_score": round(s_bm25, 4),
                "minilm_score": round(s_minilm, 4),
                "mpnet_score": round(s_mpnet, 4),
                "final_score": round(final_score, 4),
            }
        )

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    if not ranked:
        return None, 0.0, []

    best = ranked[0]
    return best["candidate"], best["final_score"], ranked


def hybrid_ensemble_match(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    qdrant_cfg_minilm: QdrantConfig,
    qdrant_cfg_mpnet: QdrantConfig,
    source_descriptions: Optional[Dict[str, Any]] = None,
    target_descriptions: Optional[Dict[str, Any]] = None,
    top_k_dense: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Hybrid Ensemble Matching:
      - BM25 (sparse)
      - MiniLM dense (Qdrant)
      - MPNet dense (Qdrant)

    Descriptions are OPTIONAL but recommended.
    """
    if weights is None:
        weights = {"bm25": 0.25, "minilm": 0.35, "mpnet": 0.40}

    # 1) BM25 top-1 (now supports descriptions too)
    bm25_res = bm25_match(
        source_schema=source_schema,
        target_schema=target_schema,
        top_k=1,
    )

    # 2) MiniLM dense search (uses descriptions in query text)
    minilm_res = match_source_to_target_dense(
        source_schema=source_schema,
        qdrant_cfg=qdrant_cfg_minilm,
        source_descriptions=source_descriptions,
        top_k=top_k_dense,
    )

    # 3) MPNet dense match (uses descriptions in indexing + query)
    mpnet_res = mpnet_dense_match(
        source_schema=source_schema,
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg_mpnet,
        source_descriptions=source_descriptions,
        target_descriptions=target_descriptions,
        top_k=top_k_dense,
        recreate_index=False,
    )

    combined = _collect_candidates(bm25_res, minilm_res, mpnet_res)

    final_matches = []

    for src, cand_map in combined.items():
        best_target, best_score, ranked_candidates = _pick_best_candidate(
            candidate_map=cand_map,
            weights=weights,
        )

        final_matches.append(
            {
                "source": src,
                "best_match": best_target,
                "confidence": round(best_score, 4),
                "match_source": "ensemble",
                "candidates": ranked_candidates,
            }
        )

    return {
        "method": "hybrid_ensemble",
        "weights": weights,
        "match_count": len(final_matches),
        "matches": final_matches,
        "debug": {
            "bm25_top1": True,
            "dense_top_k": top_k_dense,
            "minilm_collection": qdrant_cfg_minilm.collection_name,
            "mpnet_collection": qdrant_cfg_mpnet.collection_name,
            "descriptions_used": bool(source_descriptions and target_descriptions),
        },
    }
