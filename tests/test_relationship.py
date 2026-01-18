from schema_matching_toolkit import DBConfig, extract_schema
from schema_matching_toolkit.relationship_detector import detect_relationships

if __name__ == "__main__":
    cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="GPC",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    schema_data = extract_schema(cfg)
    rels = detect_relationships(cfg, schema_data)

    print(" Relationships detected")
    print("Relationships:", rels["relationship_count"])

    # print first table edges
    for t in rels["tables"]:
        if t["edges"]:
            print("Table:", t["table_name"])
            print("Edges:", t["edges"])
            break
