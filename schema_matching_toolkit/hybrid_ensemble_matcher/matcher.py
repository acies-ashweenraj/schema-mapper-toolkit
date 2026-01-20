from typing import Dict, Any, List, Tuple, Optional

from schema_matching_toolkit.sparse_bm25 import bm25_match
from schema_matching_toolkit.minilm_dense_matcher import match_source_to_target_dense
from schema_matching_toolkit.mpnet_embedding_matcher import mpnet_dense_match
from schema_matching_toolkit.common.db_config import QdrantConfig

from .table_mapper import build_table_matches_from_column_matches


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
    return best["candidate"], float(best["final_score"]), ranked


def _group_column_matches_by_table(
    column_matches: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups column matches by source table name.

    Returns:
      {
        "source_table": [col_match1, col_match2, ...]
      }
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for m in column_matches:
        src = m.get("source", "")
        if "." not in src:
            continue

        src_table = src.split(".", 1)[0]
        grouped.setdefault(src_table, []).append(m)

    return grouped


def hybrid_ensemble_match(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    qdrant_cfg_minilm: QdrantConfig,
    qdrant_cfg_mpnet: QdrantConfig,
    source_descriptions: Optional[Dict[str, Any]] = None,
    target_descriptions: Optional[Dict[str, Any]] = None,
    top_k_dense: int = 5,
    weights: Optional[Dict[str, float]] = None,
    include_table_matches: bool = True,
) -> Dict[str, Any]:
    """
    Hybrid Ensemble Matching:
      - BM25 (sparse)
      - MiniLM dense (Qdrant)
      - MPNet dense (Qdrant)

    Output format:
      - table matches first
      - inside each table -> column matches
    """

    if weights is None:
        weights = {"bm25": 0.25, "minilm": 0.35, "mpnet": 0.40}

    # 1) BM25
    bm25_res = bm25_match(
        source_schema=source_schema,
        target_schema=target_schema,
        top_k=1,
    )

    # 2) MiniLM
    minilm_res = match_source_to_target_dense(
        source_schema=source_schema,
        qdrant_cfg=qdrant_cfg_minilm,
        source_descriptions=source_descriptions,
        top_k=top_k_dense,
    )

    # 3) MPNet
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

    # -------------------------
    # Column matches
    # -------------------------
    column_matches: List[Dict[str, Any]] = []

    for src, cand_map in combined.items():
        best_target, best_score, ranked_candidates = _pick_best_candidate(
            candidate_map=cand_map,
            weights=weights,
        )

        column_matches.append(
            {
                "source": src,
                "best_match": best_target,
                "confidence": round(best_score, 4),
                "match_source": "ensemble",
                "candidates": ranked_candidates,
            }
        )

    # Always keep column count available
    out: Dict[str, Any] = {
        "column_match_count": len(column_matches),
    }

    # -------------------------
    # Table matches first + nested column matches
    # -------------------------
    if include_table_matches:
        table_info = build_table_matches_from_column_matches({"matches": column_matches})
        table_matches = table_info.get("table_matches", [])

        grouped_cols = _group_column_matches_by_table(column_matches)

        tables_out = []
        for t in table_matches:
            src_table = t.get("source_table")
            tables_out.append(
                {
                    "source_table": src_table,
                    "best_match_table": t.get("best_match_table"),
                    "confidence": t.get("confidence"),
                    "column_match_count": t.get("column_match_count", 0),
                    "column_matches": grouped_cols.get(src_table, []),
                }
            )

        out["table_match_count"] = len(tables_out)
        out["tables"] = tables_out

    else:
        out["column_matches"] = column_matches

    return out
