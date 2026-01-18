from dataclasses import dataclass


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "master_columns"

    # qdrant vector field name
    vector_name: str = "minilm_dense"
    vector_size: int = 384
