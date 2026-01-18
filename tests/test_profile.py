from schema_matching_toolkit import DBConfig, extract_schema, profile_schema


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

    schema_data = extract_schema(cfg)
    profile = profile_schema(cfg, schema_data=schema_data, sample_size=200, top_k=5)

    print("✅ Profile generated")
    print("Tables:", len(profile["tables"]))

    # print first table first column sample
    first_table = profile["tables"][0]
    first_col = first_table["columns"][0]

    print("Example table:", first_table["table_name"])
    print("Example column profile:")
    print(first_col)
