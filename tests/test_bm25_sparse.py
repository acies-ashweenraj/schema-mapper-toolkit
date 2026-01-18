from schema_matching_toolkit import DBConfig, extract_schema, bm25_match


def main():
    source_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    target_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    source_schema = extract_schema(source_cfg)
    target_schema = extract_schema(target_cfg)

    result = bm25_match(source_schema, target_schema, top_k=5)

    print("✅ BM25 Matching Done")
    print("Total source columns:", len(result["matches"]))
    print("Sample output:", result["matches"][0])


if __name__ == "__main__":
    main()
