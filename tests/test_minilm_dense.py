from schema_matching_toolkit import DBConfig, QdrantConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.llm_description import describe_schema_with_groq
from schema_matching_toolkit.minilm_dense_matcher.indexer import index_target_schema_to_qdrant
from schema_matching_toolkit.minilm_dense_matcher.matcher import match_source_to_target_dense


def main():
    GROQ_API_KEY = ""

    src_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    tgt_cfg = DBConfig(
        db_type="postgres",
        host="localhost",
        port=5432,
        database="employee",
        username="postgres",
        password="ashween29",
        schema_name="public",
    )

    qdrant_cfg = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="minilm_columns_with_desc",
        vector_name="dense_vector",
        vector_size=384,
    )

    # 1) Extract schema
    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    print("✅ Extracted source tables:", source_schema["table_count"])
    print("✅ Extracted target tables:", target_schema["table_count"])

    # 2) Describe schemas (Groq)
    groq_cfg = GroqConfig(api_key=GROQ_API_KEY)
    source_desc = describe_schema_with_groq(source_schema, groq_cfg)
    target_desc = describe_schema_with_groq(target_schema, groq_cfg)

    print("✅ Descriptions added")
    print("Source column desc:", len(source_desc.get("columns", [])))
    print("Target column desc:", len(target_desc.get("columns", [])))

    # 3) Index target schema into Qdrant
    index_info = index_target_schema_to_qdrant(
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg,
        descriptions=target_desc,
        recreate=True,
    )

    print("✅ Indexed target into Qdrant:", index_info)

    # 4) Match source → target
    result = match_source_to_target_dense(
        source_schema=source_schema,
        qdrant_cfg=qdrant_cfg,
        source_descriptions=source_desc,
        top_k=3,
    )

    print("\n✅ MiniLM Dense Matching Done")
    print("Matches:", result["match_count"])

    for m in result["matches"][:3]:
        print("\nSOURCE:", m["source"])
        print("BEST:", m["best_match"], "score:", m["confidence"])
        print("CANDIDATES:", m["candidates"])


if __name__ == "__main__":
    main()
