import json
from schema_matching_toolkit import DBConfig, QdrantConfig, GroqConfig
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
            database="employee",
            username="postgres",
            password="ashween29",
            schema_name="public",
        ),
        qdrant_cfg_minilm=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="minilm_columns",
            vector_name="dense_vector",
            vector_size=384,
        ),
        qdrant_cfg_mpnet=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="mpnet_columns",
            vector_name="dense_vector",
            vector_size=768,
        ),
        groq_cfg=GroqConfig(api_key=""),
        top_k_dense=1
        # weights={"bm25": 0.25, "minilm": 0.35, "mpnet": 0.40},
    )

    output_file = "hybrid_mapping_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved output to:", output_file)

if __name__ == "__main__":
    main()
