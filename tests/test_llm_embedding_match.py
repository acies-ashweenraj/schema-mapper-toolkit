from schema_matching_toolkit import DBConfig, QdrantConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.mpnet_embedding_matcher import mpnet_dense_match


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

    # ✅ FIX HERE
    qdrant_cfg = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="mpnet_columns",   # ✅ correct field
        vector_name="dense_vector",
        vector_size=768,                  # ✅ MPNet vector size
    )

    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    print("✅ Source tables:", source_schema["table_count"])
    print("✅ Target tables:", target_schema["table_count"])

    result = mpnet_dense_match(
        source_schema=source_schema,
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg,
        top_k=3,
        recreate_index=True,
    )

    print("✅ Dense Matching Done")
    print("Matches:", len(result["matches"]))

    for m in result["matches"][:3]:
        print("\nSOURCE:", m["source"])
        print("BEST:", m["best_match"], "score:", m["best_score"])
        print("CANDIDATES:", m["candidates"])


if __name__ == "__main__":
    main()
