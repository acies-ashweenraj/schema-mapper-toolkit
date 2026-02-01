from schema_matching_toolkit import DBConfig, GroqConfig
from schema_matching_toolkit.schema_metadata_generator import generate_schema_metadata


def main():
    db_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="ehs_client",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    metadata = generate_schema_metadata(
        db_cfg=db_cfg,
        groq_cfg=GroqConfig(api_key=""),  
        output_format="xlxs"
    )

    print("done")
    print("Saved file:", metadata.get("saved_file"))


if __name__ == "__main__":
    main()
