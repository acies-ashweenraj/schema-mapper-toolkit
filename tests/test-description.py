from schema_matching_toolkit import DBConfig, GroqConfig, extract_schema, describe_schema_with_groq


def main():
    cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    GROQ_API_KEY = ""

    schema = extract_schema(cfg)
    print("✅ Extracted schema tables:", schema["table_count"])

    groq_cfg = GroqConfig(api_key=GROQ_API_KEY)

    descriptions = describe_schema_with_groq(schema, groq_cfg)

    print("\n✅ Descriptions Generated")
    print("Tables described:", len(descriptions.get("tables", [])))
    print("Columns described:", len(descriptions.get("columns", [])))

    print("\n================ TABLE DESCRIPTIONS ================")

    # ✅ tables is a list
    for t in descriptions.get("tables", []):
        print(f"\nTABLE: {t.get('table_name')}")
        print("DESC:", t.get("description"))

    print("\n================ COLUMN DESCRIPTIONS ================")

    # ✅ columns is also a list
    for c in descriptions.get("columns", []):
        print(f"\nCOLUMN: {c.get('column_id')}")
        print("DESC:", c.get("description"))


if __name__ == "__main__":
    main()
