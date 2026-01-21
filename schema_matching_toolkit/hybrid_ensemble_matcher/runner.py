from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from schema_matching_toolkit.common.db_config import DBConfig, QdrantConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.llm_description import describe_schema_with_groq

from schema_matching_toolkit.minilm_dense_matcher import index_target_schema_to_qdrant
from schema_matching_toolkit.mpnet_embedding_matcher import index_target_columns_mpnet

from schema_matching_toolkit.hybrid_ensemble_matcher.matcher import hybrid_ensemble_match
from schema_matching_toolkit.hybrid_ensemble_matcher.exporter import save_mapping_output


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_hybrid_mapping(
    src_cfg: DBConfig,
    tgt_cfg: DBConfig,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    groq_cfg: Optional[GroqConfig] = None,
    top_k_dense: int = 5,
    weights: Optional[Dict[str, float]] = None,
    include_table_matches: bool = True,
    output_format: str = "csv",        # ✅ user decides
    output_file: Optional[str] = None, # ✅ optional file name
) -> Dict[str, Any]:
    """
    End-to-end hybrid mapping runner.

    ✅ Auto creates qdrant configs internally
    ✅ Always recreates collections
    ✅ Saves output automatically in user requested format
    ✅ Returns summary + full results
    """

    # auto qdrant configs
    qdrant_cfg_minilm = QdrantConfig(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="minilm_columns",
        vector_name="dense_vector",
        vector_size=384,
    )

    qdrant_cfg_mpnet = QdrantConfig(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="mpnet_columns",
        vector_name="dense_vector",
        vector_size=768,
    )

    # extract schemas
    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    # descriptions (NOT optional in your requirement)
    if groq_cfg is None:
        raise ValueError("groq_cfg is required for hybrid mapping descriptions")

    source_desc = describe_schema_with_groq(source_schema, groq_cfg)
    target_desc = describe_schema_with_groq(target_schema, groq_cfg)

    # always recreate index
    index_target_schema_to_qdrant(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg_minilm,
        recreate=True,
    )

    index_target_columns_mpnet(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg_mpnet,
        recreate=True,
    )

    # run matching
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

    # save output file
    saved_file = save_mapping_output(
        result=result,
        output_format=output_format,
        output_file=output_file,
    )

    # return summary + results
    return {
        "generated_at": _now_utc_iso(),
        "output_format": output_format,
        "saved_file": saved_file,
        "table_match_count": result.get("table_match_count", 0),
        "column_match_count": result.get("column_match_count", 0),
        "result": result,  # full payload
    }
