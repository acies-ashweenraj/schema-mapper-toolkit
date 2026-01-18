from schema_matching_toolkit import (
    DBConfig,
    QdrantConfig,
    extract_schema,
    index_target_schema_to_qdrant,
    match_source_to_target_dense,
)


def main():
    # Source DB
    source_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    # Target DB (for now same DB)
    target_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    qdrant_cfg = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="master_columns",
    )

    source_schema = extract_schema(source_cfg)
    target_schema = extract_schema(target_cfg)

    print("✅ Source tables:", source_schema["table_count"])
    print("✅ Target tables:", target_schema["table_count"])

    # index target into qdrant
    info = index_target_schema_to_qdrant(target_schema, qdrant_cfg)
    print("✅ Indexed:", info)

    # match source → target
    matches = match_source_to_target_dense(source_schema, qdrant_cfg, top_k=3)

    print("✅ Dense Matching Done")
    print("Matches:", matches["match_count"])

    # show first 3
    for m in matches["matches"][:3]:
        print("\nSOURCE:", m["source"])
        print("BEST:", m["best_match"], "score:", m["confidence"])
        print("CANDIDATES:", m["candidates"])


if __name__ == "__main__":
    main()
