from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from schema_matching_toolkit.common.db_config import DBConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.profiling import profile_schema
from schema_matching_toolkit.relationship_detector import detect_relationships
from schema_matching_toolkit.llm_description import describe_schema_with_groq

from .exporter import save_metadata_output


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _count_total_columns(schema: Dict[str, Any]) -> int:
    return sum(len(t.get("columns", [])) for t in schema.get("tables", []))


def _build_description_maps(descriptions: Optional[Dict[str, Any]]) -> tuple[dict, dict]:
    if not descriptions:
        return {}, {}

    table_map = {t["table_name"]: t.get("description") for t in descriptions.get("tables", []) if t.get("table_name")}
    col_map = {c["column_id"]: c.get("description") for c in descriptions.get("columns", []) if c.get("column_id")}
    return table_map, col_map


def _build_profiling_map(profiling: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
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

        pmap[tname] = {"row_count": t.get("row_count", 0), "columns": colmap}

    return pmap


def _build_relationships_items(relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for t in relationships.get("tables", []):
        items.extend(t.get("edges", []))
    return items


def generate_schema_metadata(
    db_cfg: DBConfig,
    groq_cfg: Optional[GroqConfig] = None,
    profile_sample_size: int = 50,
    profile_top_k: int = 3,
    output_format: str = "csv",          # ✅ default CSV
    output_path: Optional[str] = None,   # ✅ auto file name
) -> Dict[str, Any]:
    """
    Generates schema metadata and auto-saves output (default CSV).
    Returns metadata + saved_file path.
    """

    schema = extract_schema(db_cfg)

    profiling = profile_schema(
        cfg=db_cfg,
        schema_data=schema,
        sample_size=profile_sample_size,
        top_k=profile_top_k,
    )

    relationships = detect_relationships(db_cfg, schema)

    descriptions = None
    if groq_cfg is not None:
        descriptions = describe_schema_with_groq(schema, groq_cfg)

    table_desc_map, col_desc_map = _build_description_maps(descriptions)
    profiling_map = _build_profiling_map(profiling)

    table_count = schema.get("table_count", len(schema.get("tables", [])))
    column_count = _count_total_columns(schema)
    relationship_count = int(relationships.get("relationship_count", 0))

    relationship_items = _build_relationships_items(relationships)

    final_tables: List[Dict[str, Any]] = []

    for t in schema.get("tables", []):
        table_name = t.get("table_name")
        if not table_name:
            continue

        t_prof = profiling_map.get(table_name, {})
        row_count = int(t_prof.get("row_count", 0))

        edges = []
        for rt in relationships.get("tables", []):
            if rt.get("table_name") == table_name:
                edges = rt.get("edges", [])
                break

        columns_out: List[Dict[str, Any]] = []

        for c in t.get("columns", []):
            col_name = c.get("column_name")
            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            pcol = (t_prof.get("columns") or {}).get(col_name, {})

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
            for k in ["numeric_stats", "text_stats", "date_stats", "boolean_stats"]:
                if k in pcol:
                    profiling_payload[k] = pcol[k]

            columns_out.append(
                {
                    "column_name": col_name,
                    "data_type": c.get("data_type"),
                    "kind": pcol.get("kind"),
                    "description": col_desc_map.get(col_id),
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

    # ✅ AUTO SAVE (DEFAULT CSV)
    saved_file = save_metadata_output(
        metadata=final_output,
        output_format=output_format,
        output_path=output_path,
    )

    final_output["saved_file"] = saved_file
    return final_output
