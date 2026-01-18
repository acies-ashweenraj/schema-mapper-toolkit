from schema_matching_toolkit import DBConfig, extract_schema, profile_schema

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
profile = profile_schema(cfg, schema)

print("✅ Profile generated")
print("Tables:", len(profile["tables"]))
print(profile["tables"][0]["columns"][0])
