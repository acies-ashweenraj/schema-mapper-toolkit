from typing import Dict, Any
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
You are a database metadata expert.

Your job:
Generate business-friendly descriptions for ALL tables and ALL columns.

Return ONLY valid JSON in this exact format:
{
  "tables": [
    {"table_name": "table1", "description": "..." }
  ],
  "columns": [
    {"column_id": "table1.column1", "description": "..." }
  ]
}

STRICT RULES:
1) Every description MUST be 2 to 3 full sentences (not one line).
2) Table descriptions must explain:
   - what the table represents
   - what kind of records it stores
   - how it is typically used in analytics/business workflows
3) Column descriptions must explain:
   - what the column represents
   - what type of values it stores (based on name + datatype)
   - why it matters / how it is used
4) If meaning is unclear, make the best guess based on naming patterns.
5) Do NOT return markdown.
6) Do NOT add extra keys.
7) Output must be valid JSON only.
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

    # Safe JSON parsing (Groq sometimes wraps with text)
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start : end + 1])
        else:
            raise ValueError("Groq did not return valid JSON")

    tables = data.get("tables", [])
    columns = data.get("columns", [])

    if not isinstance(tables, list):
        tables = []
    if not isinstance(columns, list):
        columns = []

    return {"tables": tables, "columns": columns}
