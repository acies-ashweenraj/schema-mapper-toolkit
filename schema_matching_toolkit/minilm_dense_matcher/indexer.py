from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from schema_matching_toolkit.common.db_config import QdrantConfig


EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _build_column_text(table: str, column: Dict[str, Any]) -> str:
    col_name = column.get("column_name")
    dtype = column.get("data_type", "")
    return f"{table}.{col_name} type {dtype}"


def index_target_schema_to_qdrant(
    target_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
) -> Dict[str, Any]:
    """
    Index all target columns into Qdrant.

    Input:
      target_schema from extract_schema()

    Output:
      {"indexed_points": N, "collection": "..."}
    """
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    # create collection if missing
    if not client.collection_exists(qdrant_cfg.collection_name):
        client.create_collection(
            collection_name=qdrant_cfg.collection_name,
            vectors_config={
                qdrant_cfg.vector_name: VectorParams(
                    size=qdrant_cfg.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )

    points: List[PointStruct] = []
    pid = 1

    for t in target_schema.get("tables", []):
        table_name = t.get("table_name")
        for c in t.get("columns", []):
            col_name = c.get("column_name")
            if not table_name or not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            text = _build_column_text(table_name, c)

            vec = EMBEDDER.encode([text], normalize_embeddings=True)[0].tolist()

            points.append(
                PointStruct(
                    id=pid,
                    vector={qdrant_cfg.vector_name: vec},
                    payload={
                        "column_id": col_id,
                        "table": table_name,
                        "column": col_name,
                        "data_type": c.get("data_type", ""),
                        "text": text,
                    },
                )
            )
            pid += 1

    if points:
        client.upsert(collection_name=qdrant_cfg.collection_name, points=points)

    return {
        "collection": qdrant_cfg.collection_name,
        "indexed_points": len(points),
    }
