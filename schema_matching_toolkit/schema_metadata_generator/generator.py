from __future__ import annotations

from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone

from schema_matching_toolkit.common.db_config import DBConfig, GroqConfig
from schema_matching_toolkit.schema_extractor import extract_schema
from schema_matching_toolkit.profiling import profile_schema
from schema_matching_toolkit.relationship_detector import detect_relationships
from schema_matching_toolkit.llm_description import describe_schema_with_groq

from .exporter import save_metadata_output


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_description_maps(descriptions: Optional[Dict[str, Any]]) -> tuple[dict, dict]:
    if not descriptions:
        return {}, {}

    return (
        {t["table_name"]: t.get("description") for t in descriptions.get("tables", [])},
        {c["column_id"]: c.get("description") for c in descriptions.get("columns", [])},
    )


def _build_profiling_map(profiling: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pmap = {}

    for t in profiling.get("tables", []):
        tname = t.get("table_name")
        if not tname:
            continue

        pmap[tname] = {
            "row_count": int(t.get("row_count", 0)),
            "columns": {
                c.get("column").strip(): c
                for c in t.get("columns", [])
                if c.get("column")
            },
        }

    return pmap


def _build_relationship_items(relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        e
        for t in relationships.get("tables", [])
        for e in t.get("edges", [])
    ]


def _build_fk_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        f"{r['fk_table']}.{r['fk_column'].strip()}": {
            "is_foreign_key": True,
            "ref_table": r.get("pk_table"),
            "ref_column": r.get("pk_column"),
            "relationship_type": "foreign_key",
        }
        for r in items
        if r.get("fk_table") and r.get("fk_column")
    }


def _collect_pk_from_relationships(items: List[Dict[str, Any]]) -> Set[str]:
    return {
        f"{r['pk_table']}.{r['pk_column'].strip()}"
        for r in items
        if r.get("pk_table") and r.get("pk_column")
    }


def _is_primary_key(
    table_name: str,
    col_name: str,
    profiling: Dict[str, Any],
    data_type: Optional[str],
    pk_from_relationships: Set[str],
) -> bool:
    col_name = col_name.strip()
    col_id = f"{table_name}.{col_name}"
    lname = col_name.lower()

    # 1️⃣ Hard naming rule
    if lname.endswith("_pk"):
        return True

    # 2️⃣ Referenced by FK
    if col_id in pk_from_relationships:
        return True

    # 3️⃣ _id heuristic
    if lname.endswith("_id") and not lname.endswith("_fk"):
        return True

    # 4️⃣ Profiling uniqueness
    not_null = profiling.get("not_null_count", 0)
    distinct = profiling.get("distinct_count", 0)

    if not_null > 0:
        ratio = distinct / not_null
        if ratio >= 0.98 and data_type and data_type.lower() in {
            "integer", "bigint", "uuid"
        }:
            return True

    return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def generate_schema_metadata(
    db_cfg: DBConfig,
    groq_cfg: Optional[GroqConfig] = None,
    profile_sample_size: int = 50,
    profile_top_k: int = 3,
    output_format: str = "csv",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:

    schema = extract_schema(db_cfg)

    profiling = profile_schema(
        cfg=db_cfg,
        schema_data=schema,
        sample_size=profile_sample_size,
        top_k=profile_top_k,
    )

    relationships = detect_relationships(db_cfg, schema)
    relationship_items = _build_relationship_items(relationships)

    descriptions = (
        describe_schema_with_groq(schema, groq_cfg)
        if groq_cfg else None
    )

    table_desc_map, col_desc_map = _build_description_maps(descriptions)
    profiling_map = _build_profiling_map(profiling)
    pk_from_relationships = _collect_pk_from_relationships(relationship_items)
    fk_map = _build_fk_map(relationship_items)

    tables_out = []

    for t in schema.get("tables", []):
        tname = t.get("table_name")
        if not tname:
            continue

        tprof = profiling_map.get(tname, {})
        columns_out = []

        for c in t.get("columns", []):
            cname = c.get("column_name")
            if not cname:
                continue

            cname = cname.strip()
            col_id = f"{tname}.{cname}"
            pcol = (tprof.get("columns") or {}).get(cname, {})
            rel = fk_map.get(col_id, {})

            is_pk = _is_primary_key(
                tname,
                cname,
                pcol,
                c.get("data_type"),
                pk_from_relationships,
            )

            columns_out.append(
                {
                    "column_name": cname,
                    "data_type": c.get("data_type"),
                    "kind": c.get("kind"),
                    "description": col_desc_map.get(col_id),

                    # ✅ single source of truth
                    "is_primary_key": bool(is_pk),
                    "is_foreign_key": bool(rel.get("is_foreign_key", False)),
                    "ref_table": rel.get("ref_table"),
                    "ref_column": rel.get("ref_column"),
                    "relationship_type": rel.get("relationship_type"),

                    "profiling": pcol,
                }
            )

        tables_out.append(
            {
                "table_name": tname,
                "description": table_desc_map.get(tname),
                "row_count": tprof.get("row_count", 0),
                "column_count": len(columns_out),
                "columns": columns_out,
            }
        )

    metadata = {
        "generated_at": _now_utc_iso(),
        "database": vars(db_cfg),
        "summary": {
            "table_count": len(tables_out),
            "column_count": sum(len(t["columns"]) for t in tables_out),
            "relationship_count": len(relationship_items),
        },
        "tables": tables_out,
        "relationships": {
            "relationship_count": len(relationship_items),
            "items": relationship_items,
        },
    }

    metadata["saved_file"] = save_metadata_output(
        metadata, output_format, output_path
    )

    return metadata
