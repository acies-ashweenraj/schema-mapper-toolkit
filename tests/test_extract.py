from schema_matching_toolkit import DBConfig, extract_schema

if __name__ == "__main__":
    cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    result = extract_schema(cfg)

    print("✅ Extracted schema")
    print("Tables:", result["table_count"])
    print(result)
