from typing import Dict, Any, Optional

from schema_matching_toolkit.common.db_config import DBConfig, QdrantConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.llm_description import describe_schema_with_groq

from schema_matching_toolkit.minilm_dense_matcher import index_target_schema_to_qdrant
from schema_matching_toolkit.mpnet_embedding_matcher import index_target_columns_mpnet

from schema_matching_toolkit.hybrid_ensemble_matcher.matcher import hybrid_ensemble_match


def run_hybrid_mapping(
    src_cfg: DBConfig,
    tgt_cfg: DBConfig,
    qdrant_cfg_minilm: QdrantConfig,
    qdrant_cfg_mpnet: QdrantConfig,
    groq_cfg: Optional[GroqConfig] = None,
    top_k_dense: int = 5,
    weights: Optional[Dict[str, float]] = None,
    recreate_indexes: bool = True,
    include_table_matches: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end runner:
      - Extract schema
      - (Optional) Groq descriptions
      - Index target for MiniLM + MPNet
      - Hybrid ensemble match
      - Returns table matches first, then column matches inside tables
    """

    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    source_desc = None
    target_desc = None

    if groq_cfg:
        source_desc = describe_schema_with_groq(source_schema, groq_cfg)
        target_desc = describe_schema_with_groq(target_schema, groq_cfg)

    # Index target schema for MiniLM
    index_target_schema_to_qdrant(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg_minilm,
        recreate=recreate_indexes,
    )

    # Index target schema for MPNet
    index_target_columns_mpnet(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg_mpnet,
        recreate=recreate_indexes,
    )

    result = hybrid_ensemble_match(
        source_schema=source_schema,
        target_schema=target_schema,
        qdrant_cfg_minilm=qdrant_cfg_minilm,
        qdrant_cfg_mpnet=qdrant_cfg_mpnet,
        source_descriptions=source_desc,
        target_descriptions=target_desc,
        top_k_dense=top_k_dense,
        weights=weights,
        include_table_matches=include_table_matches,
    )

    return result
