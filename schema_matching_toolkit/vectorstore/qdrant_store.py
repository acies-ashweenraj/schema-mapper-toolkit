from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams

from schema_mapper_toolkit.common.db_config import QdrantConfig
from schema_mapper_toolkit.common.exceptions import QdrantError
from sentence_transformers import SentenceTransformer


EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def ensure_collection(cfg: QdrantConfig):
    try:
        client = QdrantClient(host=cfg.host, port=cfg.port)

        existing = [c.name for c in client.get_collections().collections]
        if cfg.collection in existing:
            return client

        client.create_collection(
            collection_name=cfg.collection,
            vectors_config={
                cfg.dense_vector_name: VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                cfg.sparse_vector_name: SparseVectorParams()
            },
        )
        return client

    except Exception as e:
        raise QdrantError(f"Failed to create/ensure Qdrant collection: {e}")


def index_columns(cfg: QdrantConfig, columns: List[Dict[str, Any]]):
    """
    columns must contain:
      - id: "table.column"
      - text: string
      - type: datatype
    """
    try:
        client = ensure_collection(cfg)

        texts = [c["text"] for c in columns]
        vectors = EMBEDDER.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for i, c in enumerate(columns):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector={cfg.dense_vector_name: vectors[i]},
                    payload={
                        "column_name": c["id"],
                        "type": c.get("type", ""),
                        "text": c["text"],
                    },
                )
            )

        client.upsert(collection_name=cfg.collection, points=points)
        return {"indexed": len(points), "collection": cfg.collection}

    except Exception as e:
        raise QdrantError(f"Indexing failed: {e}")
