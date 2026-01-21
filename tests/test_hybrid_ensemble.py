from schema_matching_toolkit import DBConfig, GroqConfig
from schema_matching_toolkit.hybrid_ensemble_matcher import run_hybrid_mapping


def main():
    result = run_hybrid_mapping(
        src_cfg=DBConfig(
            db_type="postgres",
            host="localhost",
            port=5432,
            database="employee",
            username="postgres",
            password="ashween29",
            schema_name="public",
        ),
        tgt_cfg=DBConfig(
            db_type="postgres",
            host="localhost",
            port=5432,
            database="GPC",
            username="postgres",
            password="ashween29",
            schema_name="public",
        ),
        qdrant_host="localhost",
        qdrant_port=6333,
        groq_cfg=GroqConfig(api_key=""),  # use env variable if needed             # json / csv / xlsx
        top_k_dense=5,
    )

    print("Done")
    print("Generated at:", result["generated_at"])
    print("Saved file:", result["saved_file"])
    print("Tables matched:", result["table_match_count"])
    print("Columns matched:", result["column_match_count"])


if __name__ == "__main__":
    main()
