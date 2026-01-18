from typing import Dict, Any, List
import json

from groq import Groq
from schema_matching_toolkit.common.db_config import GroqConfig


def _schema_to_prompt(schema: Dict[str, Any]) -> str:
    """
    Convert extracted schema into a compact text prompt.
    """
    lines = []
    for t in schema.get("tables", []):
        table_name = t.get("table_name") or t.get("table") or t.get("name")
        if not table_name:
            continue

        lines.append(f"TABLE: {table_name}")
        for c in t.get("columns", []):
            col_name = c.get("column_name") or c.get("name") or c.get("column")
            dtype = c.get("data_type") or ""
            if col_name:
                lines.append(f"  - {col_name} ({dtype})")

    return "\n".join(lines)


def describe_schema_with_groq(
    schema: Dict[str, Any],
    groq_cfg: GroqConfig,
) -> Dict[str, Any]:
    """
    Input:
      schema: output of extract_schema()
      groq_cfg: GroqConfig(api_key="...")

    Output:
      {
        "tables": [{"table_name": "...", "description": "..."}],
        "columns": [{"column_id": "table.col", "description": "..."}]
      }
    """

    client = Groq(api_key=groq_cfg.api_key)

    schema_text = _schema_to_prompt(schema)

    system_prompt = """
You are a metadata expert.
Generate business-friendly descriptions for database tables and columns.

Return ONLY valid JSON in this format:

{
  "tables": [
    {"table_name": "table1", "description": "..." }
  ],
  "columns": [
    {"column_id": "table1.column1", "description": "..." }
  ]
}

Rules:
- Use short but meaningful descriptions
- If unclear, make best guess based on name + datatype
- Do NOT return markdown
- Do NOT include extra keys
"""

    user_prompt = f"""
Here is the database schema:

{schema_text}

Now generate JSON descriptions for ALL tables and ALL columns.
"""

    resp = client.chat.completions.create(
        model=groq_cfg.model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    # ✅ Safe JSON parsing (Groq sometimes wraps with text)
    try:
        data = json.loads(raw)
    except Exception:
        # try to extract JSON block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start : end + 1])
        else:
            raise ValueError("Groq did not return valid JSON")

    # Ensure structure exists
    tables = data.get("tables", [])
    columns = data.get("columns", [])

    if not isinstance(tables, list):
        tables = []
    if not isinstance(columns, list):
        columns = []

    return {"tables": tables, "columns": columns}
