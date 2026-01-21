from __future__ import annotations

import json
from typing import Dict, Any, List
from datetime import datetime, timezone


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _flatten_mapping_for_csv(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert nested output into flat rows for CSV/XLSX.
    Each row = one column match.
    """
    rows: List[Dict[str, Any]] = []

    tables = result.get("tables", [])
    for t in tables:
        src_table = t.get("source_table")
        tgt_table = t.get("best_match_table")
        t_conf = t.get("confidence")

        for cm in t.get("column_matches", []):
            rows.append(
                {
                    "source_table": src_table,
                    "target_table": tgt_table,
                    "table_confidence": t_conf,
                    "source_column": cm.get("source"),
                    "best_match_column": cm.get("best_match"),
                    "column_confidence": cm.get("confidence"),
                }
            )

    return rows


def save_mapping_output(
    result: Dict[str, Any],
    output_format: str = "csv",
    output_file: str | None = None,
) -> str:
    """
    Saves mapping output in json/csv/xlsx format.
    Returns saved file path.
    """
    output_format = (output_format or "csv").lower().strip()

    if output_format not in {"json", "csv", "xlsx"}:
        raise ValueError("output_format must be one of: json, csv, xlsx")

    # default file name
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hybrid_mapping_output_{ts}.{output_format}"

    # JSON
    if output_format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return output_file

    # CSV / XLSX needs flattening
    rows = _flatten_mapping_for_csv(result)

    # CSV
    if output_format == "csv":
        import csv

        if not rows:
            # still create empty csv with headers
            headers = [
                "source_table",
                "target_table",
                "table_confidence",
                "source_column",
                "best_match_column",
                "column_confidence",
            ]
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
            return output_file

        headers = list(rows[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        return output_file

    # XLSX
    if output_format == "xlsx":
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "mapping"

        headers = [
            "source_table",
            "target_table",
            "table_confidence",
            "source_column",
            "best_match_column",
            "column_confidence",
        ]
        ws.append(headers)

        for r in rows:
            ws.append([r.get(h) for h in headers])

        wb.save(output_file)
        return output_file

    return output_file
