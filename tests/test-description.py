from schema_matching_toolkit import DBConfig, GroqConfig, extract_schema, describe_schema_with_groq


def main():
    # 1) Extract schema
    cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    schema = extract_schema(cfg)
    print("✅ Extracted schema tables:", schema["table_count"])

    # 2) Groq config (only API key needed)
    groq_cfg = GroqConfig(
        api_key=""
    )

    # 3) Generate descriptions
    descriptions = describe_schema_with_groq(schema, groq_cfg)

    print("\n✅ Descriptions Generated")
    print("Tables described:", len(descriptions.get("tables", {})))
    print("Columns described:", len(descriptions.get("columns", {})))

    # 4) Print output nicely
    print("\n================ TABLE DESCRIPTIONS ================")
    for table_name, desc in descriptions.get("tables", {}).items():
        print(f"\n📌 {table_name}")
        print(desc)

    print("\n================ COLUMN DESCRIPTIONS ================")
    for col_id, desc in descriptions.get("columns", {}).items():
        print(f"\n🔹 {col_id}")
        print(desc)


if __name__ == "__main__":
    main()
