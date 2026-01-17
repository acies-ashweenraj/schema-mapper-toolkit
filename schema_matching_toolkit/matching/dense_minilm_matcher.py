from typing import Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from schema_mapper_toolkit.common.db_config import QdrantConfig


EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def dense_topk(cfg: QdrantConfig, query_text: str, top_k: int) -> Dict[str, float]:
    client = QdrantClient(host=cfg.host, port=cfg.port)

    qvec = EMBEDDER.encode([query_text], normalize_embeddings=True)[0].tolist()

    hits = client.search(
        collection_name=cfg.collection,
        query_vector=(cfg.dense_vector_name, qvec),
        limit=top_k,
        with_payload=True,
    )

    out = {}
    for h in hits:
        payload = h.payload or {}
        col_name = payload.get("column_name")
        if col_name:
            out[col_name] = float(h.score)
    return out
