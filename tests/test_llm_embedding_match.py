from schema_matching_toolkit import DBConfig, QdrantConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.llm_description import describe_schema_with_groq
from schema_matching_toolkit.mpnet_embedding_matcher import mpnet_dense_match


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
        collection_name="mpnet_columns_with_desc",
        vector_name="dense_vector",
        vector_size=768,  # ✅ MPNet size
    )

    source_schema = extract_schema(src_cfg)
    target_schema = extract_schema(tgt_cfg)

    print("✅ Source tables:", source_schema["table_count"])
    print("✅ Target tables:", target_schema["table_count"])

    groq_cfg = GroqConfig(api_key=GROQ_API_KEY)

    source_desc = describe_schema_with_groq(source_schema, groq_cfg)
    target_desc = describe_schema_with_groq(target_schema, groq_cfg)

    print("✅ Groq descriptions added")
    print("Source columns described:", len(source_desc.get("columns", [])))
    print("Target columns described:", len(target_desc.get("columns", [])))

    result = mpnet_dense_match(
        source_schema=source_schema,
        target_schema=target_schema,
        qdrant_cfg=qdrant_cfg,
        source_descriptions=source_desc,
        target_descriptions=target_desc,
        top_k=3,
        recreate_index=True,
    )

    print("\n✅ MPNet Dense Matching Done")
    print("Matches:", len(result["matches"]))

    for m in result["matches"][:3]:
        print("\nSOURCE:", m["source"])
        print("BEST:", m["best_match"], "score:", m["best_score"])
        print("CANDIDATES:", m["candidates"])


if __name__ == "__main__":
    main()
