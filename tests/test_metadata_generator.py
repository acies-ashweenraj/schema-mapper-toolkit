import json

from schema_matching_toolkit import DBConfig, GroqConfig
from schema_matching_toolkit.schema_metadata_generator import generate_schema_metadata


def main():
    GROQ_API_KEY = "" 
    groq_cfg = GroqConfig(api_key=GROQ_API_KEY)

    db_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="GPC",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    metadata = generate_schema_metadata(
        db_cfg=db_cfg,
        groq_cfg=groq_cfg,
        profile_sample_size=200,
        profile_top_k=5,
    )

    # save output
    output_file = "schema_metadata_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
