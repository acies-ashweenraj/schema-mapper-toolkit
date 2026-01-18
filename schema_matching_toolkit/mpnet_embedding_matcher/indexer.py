from typing import Dict, Any, List
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from schema_matching_toolkit.common.db_config import QdrantConfig


# MPNet embedder (768 dim)
MPNET = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _flatten_columns(schema: Dict[str, Any]) -> List[Dict[str, str]]:
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

            cols.append(
                {
                    "id": col_id,
                    "text": text,
                    "data_type": dtype,
                }
            )

    return cols


def index_target_columns_mpnet(
    target_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
    recreate: bool = True,
) -> Dict[str, Any]:
    """
    Index target columns into Qdrant using MPNet embeddings.
    """
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    target_cols = _flatten_columns(target_schema)

    if recreate:
        client.recreate_collection(
            collection_name=qdrant_cfg.collection_name,   # ✅ FIXED
            vectors_config={
                qdrant_cfg.vector_name: VectorParams(
                    size=qdrant_cfg.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )

    texts = [c["text"] for c in target_cols]
    vectors = MPNET.encode(texts, normalize_embeddings=True)

    points = []
    for i, col in enumerate(target_cols):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={qdrant_cfg.vector_name: vectors[i].tolist()},
                payload={
                    "column_name": col["id"],
                    "data_type": col["data_type"],
                    "text": col["text"],
                },
            )
        )

    if points:
        client.upsert(collection_name=qdrant_cfg.collection_name, points=points)

    return {
        "collection": qdrant_cfg.collection_name,
        "indexed_points": len(points),
        "vector_name": qdrant_cfg.vector_name,
        "vector_size": qdrant_cfg.vector_size,
    }
