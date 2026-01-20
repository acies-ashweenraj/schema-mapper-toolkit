from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


def _timestamp(prefix: str) -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def _flatten_metadata_to_rows(metadata: Dict[str, Any]) -> list:
    rows = []

    db = metadata.get("database", {})
    summary = metadata.get("summary", {})

    for t in metadata.get("tables", []):
        table_name = t.get("table_name")

        for c in t.get("columns", []):
            prof = c.get("profiling", {}) or {}

            rows.append(
                {
                    "db_type": db.get("db_type"),
                    "host": db.get("host"),
                    "port": db.get("port"),
                    "database": db.get("database"),
                    "schema_name": db.get("schema_name"),
                    "table_count": summary.get("table_count"),
                    "column_count": summary.get("column_count"),
                    "relationship_count": summary.get("relationship_count"),
                    "table_name": table_name,
                    "table_description": t.get("description"),
                    "table_row_count": t.get("row_count"),
                    "column_name": c.get("column_name"),
                    "data_type": c.get("data_type"),
                    "kind": c.get("kind"),
                    "column_description": c.get("description"),
                    "row_count": prof.get("row_count"),
                    "not_null_count": prof.get("not_null_count"),
                    "null_count": prof.get("null_count"),
                    "null_percent": prof.get("null_percent"),
                    "distinct_count": prof.get("distinct_count"),
                    "distinct_percent": prof.get("distinct_percent"),
                    "duplicate_count": prof.get("duplicate_count"),
                    "entropy": prof.get("entropy"),
                    "top_values": json.dumps(prof.get("top_values", []), ensure_ascii=False),
                    "sample_values": json.dumps(prof.get("sample_values", []), ensure_ascii=False),
                    "numeric_stats": json.dumps(prof.get("numeric_stats", {}), ensure_ascii=False),
                    "text_stats": json.dumps(prof.get("text_stats", {}), ensure_ascii=False),
                    "date_stats": json.dumps(prof.get("date_stats", {}), ensure_ascii=False),
                    "boolean_stats": json.dumps(prof.get("boolean_stats", {}), ensure_ascii=False),
                }
            )

    return rows


def save_metadata_output(
    metadata: Dict[str, Any],
    output_format: str = "csv",
    output_path: Optional[str] = None,
) -> str:
    """
    Default save = CSV
    Returns saved file path
    """
    output_format = (output_format or "csv").lower().strip()

    if not output_path:
        base = _timestamp("schema_metadata")
        if output_format == "json":
            output_path = f"{base}.json"
        elif output_format == "xlsx":
            output_path = f"{base}.xlsx"
        else:
            output_format = "csv"
            output_path = f"{base}.csv"

    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # JSON
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return output_path

    # CSV / XLSX
    import pandas as pd

    rows = _flatten_metadata_to_rows(metadata)
    df = pd.DataFrame(rows)

    if output_format == "xlsx":
        df.to_excel(output_path, index=False)
        return output_path

    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path
