# from typing import Dict, Any, Optional

# from schema_matching_toolkit.common.db_config import DBConfig, GroqConfig
# from schema_matching_toolkit.schema_extractor import extract_schema
# from schema_matching_toolkit.profiling import profile_schema
# from schema_matching_toolkit.relationship_detector import detect_relationships
# from schema_matching_toolkit.llm_description import describe_schema_with_groq


# def _merge_descriptions_into_schema(
#     schema: Dict[str, Any],
#     descriptions: Dict[str, Any],
# ) -> Dict[str, Any]:
#     table_desc_map = {
#         t["table_name"]: t.get("description")
#         for t in descriptions.get("tables", [])
#         if "table_name" in t
#     }

#     col_desc_map = {
#         c["column_id"]: c.get("description")
#         for c in descriptions.get("columns", [])
#         if "column_id" in c
#     }

#     for t in schema.get("tables", []):
#         table_name = t.get("table_name")
#         if table_name:
#             t["description"] = table_desc_map.get(table_name)

#         for col in t.get("columns", []):
#             col_name = col.get("column_name")
#             if table_name and col_name:
#                 col_id = f"{table_name}.{col_name}"
#                 col["description"] = col_desc_map.get(col_id)

#     return schema


# def generate_schema_metadata(
#     db_cfg: DBConfig,
#     groq_cfg: Optional[GroqConfig] = None,
#     profile_sample_size: int = 500,
#     profile_top_k: int = 10,
# ) -> Dict[str, Any]:
#     """
#     Generates final schema metadata:
#       - extracted schema
#       - profiling metrics
#       - relationship detection
#       - optional Groq descriptions (2-3 sentences)
#     """

#     schema = extract_schema(db_cfg)

#     profiling = profile_schema(
#         cfg=db_cfg,
#         schema_data=schema,
#         sample_size=profile_sample_size,
#         top_k=profile_top_k,
#     )

#     # ✅ FIX: detect_relationships needs (cfg, schema_data)
#     relationships = detect_relationships(db_cfg, schema)


#     descriptions = None
#     if groq_cfg:
#         descriptions = describe_schema_with_groq(schema, groq_cfg)
#         schema = _merge_descriptions_into_schema(schema, descriptions)

#     return {
#         "schema": schema,
#         "profiling": profiling,
#         "relationships": relationships,
#         "descriptions": descriptions,
#     }
from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from schema_matching_toolkit.common.db_config import DBConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.profiling import profile_schema
from schema_matching_toolkit.relationship_detector import detect_relationships
from schema_matching_toolkit.llm_description import describe_schema_with_groq


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _count_total_columns(schema: Dict[str, Any]) -> int:
    total = 0
    for t in schema.get("tables", []):
        total += len(t.get("columns", []))
    return total


def _build_description_maps(descriptions: Optional[Dict[str, Any]]) -> tuple[dict, dict]:
    """
    Converts:
      {
        "tables": [{"table_name":..., "description":...}],
        "columns": [{"column_id":"table.col", "description":...}]
      }
    into fast lookup dicts.
    """
    if not descriptions:
        return {}, {}

    table_map = {}
    for t in descriptions.get("tables", []):
        tn = t.get("table_name")
        if tn:
            table_map[tn] = t.get("description")

    col_map = {}
    for c in descriptions.get("columns", []):
        cid = c.get("column_id")
        if cid:
            col_map[cid] = c.get("description")

    return table_map, col_map


def _build_profiling_map(profiling: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert profiling output into:
    {
      "table": {
         "row_count": ...,
         "columns": {
             "col": { ...profiling metrics... }
         }
      }
    }
    """
    pmap: Dict[str, Dict[str, Any]] = {}

    for t in profiling.get("tables", []):
        tname = t.get("table_name")
        if not tname:
            continue

        colmap = {}
        for c in t.get("columns", []):
            cname = c.get("column")
            if cname:
                colmap[cname] = c

        pmap[tname] = {
            "row_count": t.get("row_count", 0),
            "columns": colmap,
        }

    return pmap


def _build_relationships_items(relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    detect_relationships() returns:
      {
        "schema_name": ...,
        "tables": [{table_name, edges:[...]}, ...],
        "relationship_count": ...,
        "relationship_method": ...
      }

    We flatten all edges into a single list for global "relationships.items".
    """
    items: List[Dict[str, Any]] = []

    for t in relationships.get("tables", []):
        for e in t.get("edges", []):
            if isinstance(e, dict):
                items.append(e)

    return items


def generate_schema_metadata(
    db_cfg: DBConfig,
    groq_cfg: Optional[GroqConfig] = None,
    profile_sample_size: int = 500,
    profile_top_k: int = 10,
) -> Dict[str, Any]:
    """
    Final Metadata Generator

    Output format:
    {
      "generated_at": "...",
      "database": {...},
      "summary": {...},
      "tables": [...],
      "relationships": {...}
    }
    """

    # 1) Extract schema
    schema = extract_schema(db_cfg)

    # 2) Profiling (uses schema_data)
    profiling = profile_schema(
        cfg=db_cfg,
        schema_data=schema,
        sample_size=profile_sample_size,
        top_k=profile_top_k,
    )

    # 3) Relationships
    relationships = detect_relationships(db_cfg, schema)

    # 4) Groq descriptions (optional)
    descriptions = None
    if groq_cfg is not None:
        descriptions = describe_schema_with_groq(schema, groq_cfg)

    table_desc_map, col_desc_map = _build_description_maps(descriptions)
    profiling_map = _build_profiling_map(profiling)

    # 5) Build final output
    table_count = schema.get("table_count", len(schema.get("tables", [])))
    column_count = _count_total_columns(schema)
    relationship_count = int(relationships.get("relationship_count", 0))

    # Flatten global relationship list
    relationship_items = _build_relationships_items(relationships)

    final_tables: List[Dict[str, Any]] = []

    for t in schema.get("tables", []):
        table_name = t.get("table_name")
        if not table_name:
            continue

        # get profiling table info
        t_prof = profiling_map.get(table_name, {})
        row_count = int(t_prof.get("row_count", 0))

        # edges from relationship detector
        edges = []
        for rt in relationships.get("tables", []):
            if rt.get("table_name") == table_name:
                edges = rt.get("edges", [])
                break

        columns_out: List[Dict[str, Any]] = []

        for c in t.get("columns", []):
            col_name = c.get("column_name")
            dtype = c.get("data_type", "")
            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"

            # attach groq description
            col_desc = col_desc_map.get(col_id)

            # attach profiling metrics
            pcol = (t_prof.get("columns") or {}).get(col_name, {})

            # clean profiling output: remove duplicate keys if you want
            profiling_payload = {
                "row_count": pcol.get("row_count"),
                "not_null_count": pcol.get("not_null_count"),
                "null_count": pcol.get("null_count"),
                "null_percent": pcol.get("null_percent"),
                "distinct_count": pcol.get("distinct_count"),
                "distinct_percent": pcol.get("distinct_percent"),
                "duplicate_count": pcol.get("duplicate_count"),
                "entropy": pcol.get("entropy"),
                "top_values": pcol.get("top_values", []),
                "sample_values": pcol.get("sample_values", []),
            }

            # include extra stats if present
            if "numeric_stats" in pcol:
                profiling_payload["numeric_stats"] = pcol["numeric_stats"]
            if "text_stats" in pcol:
                profiling_payload["text_stats"] = pcol["text_stats"]
            if "date_stats" in pcol:
                profiling_payload["date_stats"] = pcol["date_stats"]
            if "boolean_stats" in pcol:
                profiling_payload["boolean_stats"] = pcol["boolean_stats"]

            columns_out.append(
                {
                    "column_name": col_name,
                    "data_type": dtype,
                    "kind": pcol.get("kind") or None,
                    "description": col_desc,
                    "profiling": profiling_payload,
                }
            )

        final_tables.append(
            {
                "table_name": table_name,
                "description": table_desc_map.get(table_name),
                "row_count": row_count,
                "column_count": len(columns_out),
                "columns": columns_out,
                "edges": edges,
            }
        )

    final_output = {
        "generated_at": _now_utc_iso(),
        "database": {
            "db_type": db_cfg.db_type,
            "host": db_cfg.host,
            "port": db_cfg.port,
            "database": db_cfg.database,
            "schema_name": db_cfg.schema_name,
        },
        "summary": {
            "table_count": int(table_count),
            "column_count": int(column_count),
            "relationship_count": int(relationship_count),
        },
        "tables": final_tables,
        "relationships": {
            "relationship_count": int(relationship_count),
            "relationship_method": relationships.get("relationship_method"),
            "items": relationship_items,
        },
    }

    return final_output
