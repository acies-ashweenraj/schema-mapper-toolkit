from typing import Dict, Any, List
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

from schema_matching_toolkit.common.db_config import QdrantConfig


EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _get_column_description(descriptions: Dict[str, Any], col_id: str) -> str:
    """
    descriptions format:
    {
      "tables": [{"table_name": "...", "description": "..."}],
      "columns": [{"column_id": "table.col", "description": "..."}]
    }
    """
    if not descriptions:
        return ""

    cols = descriptions.get("columns", [])
    if not isinstance(cols, list):
        return ""

    for c in cols:
        if c.get("column_id") == col_id:
            return c.get("description") or ""

    return ""


def _flatten_target_columns_with_desc(
    target_schema: Dict[str, Any],
    descriptions: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Output:
    [
      {
        "column_id": "table.col",
        "text": "table col datatype description",
        "data_type": "integer"
      }
    ]
    """
    cols = []

    for t in target_schema.get("tables", []):
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

            # ✅ Embed richer text
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


def index_target_schema_to_qdrant(
    target_schema: Dict[str, Any],
    qdrant_cfg: QdrantConfig,
    descriptions: Dict[str, Any] | None = None,
    recreate: bool = True,
) -> Dict[str, Any]:
    """
    Index target schema columns into Qdrant using MiniLM embeddings.

    Input:
      target_schema = extract_schema(DBConfig)
      qdrant_cfg = QdrantConfig(...)
      descriptions = output of describe_schema_with_groq() (optional)
      recreate = True -> delete & recreate collection

    Output:
      {"collection": "...", "indexed_points": N}
    """
    client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port)

    # Flatten + include description
    cols = _flatten_target_columns_with_desc(target_schema, descriptions)

    if recreate:
        # delete if exists
        try:
            client.delete_collection(collection_name=qdrant_cfg.collection_name)
        except Exception:
            pass

        # create collection
        client.create_collection(
            collection_name=qdrant_cfg.collection_name,
            vectors_config={
                qdrant_cfg.vector_name: VectorParams(
                    size=qdrant_cfg.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )

    # Embed all target columns
    texts = [c["text"] for c in cols]
    vectors = EMBEDDER.encode(texts, normalize_embeddings=True)

    points = []
    for i, c in enumerate(cols):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={qdrant_cfg.vector_name: vectors[i].tolist()},
                payload={
                    "column_id": c["column_id"],
                    "column_name": c["column_id"],
                    "data_type": c["data_type"],
                    "description": c["description"],
                    "text": c["text"],
                },
            )
        )

    if points:
        client.upsert(
            collection_name=qdrant_cfg.collection_name,
            points=points,
        )

    return {"collection": qdrant_cfg.collection_name, "indexed_points": len(points)}
