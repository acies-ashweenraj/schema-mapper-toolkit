from typing import Dict, Any, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from schema_matching_toolkit.common.db_config import QdrantConfig
from schema_matching_toolkit.mpnet_embedding_matcher.indexer import index_target_columns_mpnet


MPNET = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _flatten_source(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    cols = []

    for t in schema.get("tables", []):
        table_name = t.get("table_name") or t.get("table") or t.get("name")
        if not table_name:
            continue

        for c in t.get("columns", []):
            col_name = c.get("column_name") or c.get("column") or c.get("name")
            dtype = c.get("data_type") or c.get("type") or ""

            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            text = f"{table_name} {col_name} {dtype}".strip()

            cols.append({"id": col_id, "text": text})

    return cols


def mpnet_dense_match(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
    top_k: int = 3,
    recreate_index: bool = True,
) -> Dict[str, Any]:
    """
    Dense matching using MPNet embeddings + Qdrant
    """
    # 1) Index target schema
    index_info = index_target_columns_mpnet(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg,
        recreate=recreate_index,
    )

    # 2) Init qdrant client
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    # 3) Flatten source schema
    source_cols = _flatten_source(source_schema)

    matches = []

    for src in source_cols:
        qvec = MPNET.encode([src["text"]], normalize_embeddings=True)[0].tolist()

        hits = client.search(
            collection_name=qdrant_cfg.collection_name,  # ✅ FIXED
            query_vector=(qdrant_cfg.vector_name, qvec),
            limit=top_k,
            with_payload=True,
        )

        candidates = []
        for h in hits:
            payload = h.payload or {}
            candidates.append(
                {
                    "target": payload.get("column_name", str(h.id)),
                    "score": float(h.score),
                    "data_type": payload.get("data_type", ""),
                }
            )

        best_match = candidates[0]["target"] if candidates else None
        best_score = candidates[0]["score"] if candidates else 0.0

        matches.append(
            {
                "source": src["id"],
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
