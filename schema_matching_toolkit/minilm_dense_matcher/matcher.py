from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from schema_matching_toolkit.common.db_config import QdrantConfig


EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def match_source_to_target_dense(
    source_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Dense matching using MiniLM embeddings in Qdrant.

    Input:
      source_schema extracted using extract_schema()
      qdrant_cfg (host, port, collection_name)

    Output:
      {
        "match_count": N,
        "matches": [
           {
             "source": "table.col",
             "candidates": [
               {"target": "table.col", "score": 0.91}
             ]
           }
        ]
      }
    """
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    results = []

    for t in source_schema.get("tables", []):
        table_name = t.get("table_name")
        for c in t.get("columns", []):
            col_name = c.get("column_name")
            if not table_name or not col_name:
                continue

            source_id = f"{table_name}.{col_name}"
            text = f"{source_id} type {c.get('data_type', '')}"

            qvec = EMBEDDER.encode([text], normalize_embeddings=True)[0].tolist()

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
                    }
                )

            results.append(
                {
                    "source": source_id,
                    "source_type": c.get("data_type", ""),
                    "candidates": candidates,
                    "best_match": candidates[0]["target"] if candidates else None,
                    "confidence": candidates[0]["score"] if candidates else 0.0,
                }
            )

    return {"match_count": len(results), "matches": results}
