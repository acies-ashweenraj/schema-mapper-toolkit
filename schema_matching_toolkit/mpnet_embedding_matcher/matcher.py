from typing import Dict, Any, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from schema_matching_toolkit.common.db_config import QdrantConfig
from schema_matching_toolkit.mpnet_embedding_matcher.indexer import index_target_columns_mpnet


MPNET = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _get_column_description(descriptions: Dict[str, Any], col_id: str) -> str:
    if not descriptions:
        return ""

    cols = descriptions.get("columns", [])
    if not isinstance(cols, list):
        return ""

    for c in cols:
        if c.get("column_id") == col_id:
            return c.get("description") or ""

    return ""


def _flatten_source_with_desc(
    source_schema: Dict[str, Any],
    descriptions: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    cols = []

    for t in source_schema.get("tables", []):
        table_name = t.get("table_name") or t.get("table") or t.get("name")
        if not table_name:
            continue

        for c in t.get("columns", []):
            col_name = c.get("column_name") or c.get("column") or c.get("name")
            dtype = c.get("data_type") or ""

            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            desc = _get_column_description(descriptions or {}, col_id)

            text = f"{table_name} {col_name} {dtype} {desc}".strip()

            cols.append(
                {
                    "column_id": col_id,
                    "text": text,
                    "data_type": dtype,
                    "description": desc,
                }
            )

    return cols


def mpnet_dense_match(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
    source_descriptions: Dict[str, Any] | None = None,
    target_descriptions: Dict[str, Any] | None = None,
    top_k: int = 3,
    recreate_index: bool = True,
) -> Dict[str, Any]:
    """
    Dense matching using MPNet embeddings + Qdrant (with optional Groq descriptions)
    """
    # 1) Index target
    index_info = index_target_columns_mpnet(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg,
        descriptions=target_descriptions,
        recreate=recreate_index,
    )

    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    source_cols = _flatten_source_with_desc(source_schema, source_descriptions)

    matches = []

    for src in source_cols:
        qvec = MPNET.encode([src["text"]], normalize_embeddings=True)[0].tolist()

        hits = client.search(
            collection_name=qdrant_cfg.collection_name,
            query_vector=(qdrant_cfg.vector_name, qvec),
            limit=top_k,
            with_payload=True,
        )

        candidates = []
        for h in hits:
            payload = h.payload or {}
            candidates.append(
                {
                    "target": payload.get("column_id", str(h.id)),
                    "score": float(h.score),
                    "data_type": payload.get("data_type", ""),
                    "description": payload.get("description", ""),
                }
            )

        best_match = candidates[0]["target"] if candidates else None
        best_score = candidates[0]["score"] if candidates else 0.0

        matches.append(
            {
                "source": src["column_id"],
                "source_type": src["data_type"],
                "best_match": best_match,
                "best_score": best_score,
                "candidates": candidates,
            }
        )

    return {
        "method": "mpnet_dense_qdrant",
        "index_info": index_info,
        "top_k": top_k,
        "matches": matches,
    }
