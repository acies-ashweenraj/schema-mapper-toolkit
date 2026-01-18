from schema_matching_toolkit import DBConfig, QdrantConfig, extract_schema
from schema_matching_toolkit.minilm_dense_matcher import index_target_schema_to_qdrant
from schema_matching_toolkit.hybrid_ensemble_matcher import (
    hybrid_ensemble_match,
    build_table_matches_from_column_matches,
)


def main():
    src_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    tgt_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    qdrant_minilm = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="minilm_columns",
        vector_name="dense_vector",
        vector_size=384,
    )

    qdrant_mpnet = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="mpnet_columns",
        vector_name="dense_vector",
        vector_size=768,
    )

    # MiniLM target index
    index_target_schema_to_qdrant(
        target_schema=target_schema,
        qdrant_cfg=qdrant_minilm,
        recreate=True,
    )

    # Hybrid column matching
    hybrid_result = hybrid_ensemble_match(
        source_schema=source_schema,
        target_schema=target_schema,
        qdrant_cfg_minilm=qdrant_minilm,
        qdrant_cfg_mpnet=qdrant_mpnet,
        top_k_dense=5,
    )

    # Convert to table matches
    table_result = build_table_matches_from_column_matches(hybrid_result)

    print("\n================ TABLE MATCHES ================")
    for t in table_result["table_matches"]:
        print(
            f"{t['source_table']}  -->  {t['best_match_table']}  "
            f"(conf={t['confidence']}, cols={t['column_match_count']})"
        )

    print("\n================ COLUMN MATCHES (sample) ================")
    for m in table_result["column_matches"][:5]:
        print("\nSOURCE:", m["source"])
        print("BEST:", m["best_match"])
        print("CONF:", m["confidence"])
        print("TOP3:", m["candidates"][:3])


if __name__ == "__main__":
    main()
