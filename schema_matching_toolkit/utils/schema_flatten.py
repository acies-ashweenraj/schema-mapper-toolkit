from typing import Dict, Any, List


def flatten_schema_columns(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts schema into a flat list:
    [
      {
        "id": "table.column",
        "table": "table",
        "column": "column",
        "data_type": "integer",
        "text": "table column datatype"
      }
    ]
    """
    cols: List[Dict[str, Any]] = []

    for t in schema.get("tables", []):
        table_name = t.get("table_name") or t.get("table") or t.get("name")
        if not table_name:
            continue

        for c in t.get("columns", []):
            col_name = c.get("column_name") or c.get("column") or c.get("name")
            dtype = c.get("data_type") or c.get("type") or ""

            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            text = f"{table_name} {col_name} {dtype}".lower()

            cols.append(
                {
                    "id": col_id,
                    "table": table_name,
                    "column": col_name,
                    "data_type": dtype,
                    "text": text,
                }
            )

    return cols
