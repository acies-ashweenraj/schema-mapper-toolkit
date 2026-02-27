from typing import Dict, Any
import json
import re

from groq import Groq
from schema_matching_toolkit.common.db_config import GroqConfig


# ------------------------------------------------------------
# Convert schema dict → compact prompt text
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# SAFE JSON LOADER (LLM-PROOF)
# ------------------------------------------------------------
def _safe_load_llm_json(raw: str) -> Dict[str, Any]:
    """
    Safely parse JSON returned by LLMs.
    Handles:
      - extra explanations
      - trailing commas
      - smart quotes
      - hidden control chars
      - partial wrapping text
    """

    if not raw or not raw.strip():
        raise ValueError("Groq returned empty response")

    # Extract JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in Groq response:\n{raw}")

    json_str = match.group(0)

    # --- Fix common LLM mistakes ---

    # remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # replace smart quotes
    json_str = json_str.replace("“", '"').replace("”", '"')

    # remove control characters
    json_str = re.sub(r"[\x00-\x1F]+", " ", json_str)

    try:
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print("\n========== BROKEN GROQ OUTPUT ==========")
        print(json_str)
        print("========================================\n")
        raise ValueError(f"Invalid JSON returned by Groq: {e}")


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
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

    # --------------------------------------------------------
    # SYSTEM PROMPT (STRICT JSON)
    # --------------------------------------------------------
    system_prompt = """
You are a database metadata expert.

Your job:
Generate business-friendly descriptions for ALL tables and ALL columns.

Return ONLY valid JSON in this exact format:
{
  "tables": [
    {"table_name": "table1", "description": "..."}
  ],
  "columns": [
    {"column_id": "table1.column1", "description": "..."}
  ]
}

STRICT RULES:
1) Every description MUST be 2–3 full sentences.
2) Table descriptions must explain:
   - what the table represents
   - what records it stores
   - how it is used in business analytics
3) Column descriptions must explain:
   - what the column represents
   - type of values stored
   - why it matters
4) Make best guess if unclear.
5) NO markdown.
6) NO explanations.
7) NO extra keys.
8) Output must START with '{' and END with '}'.
"""

    user_prompt = f"""
Here is the database schema:

{schema_text}

Generate descriptions now.
"""

    # --------------------------------------------------------
    # GROQ CALL
    # --------------------------------------------------------
    resp = client.chat.completions.create(
        model=groq_cfg.model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=4096,   # prevents truncation errors
    )

    raw = resp.choices[0].message.content or ""
    raw = raw.strip()

    # Debug (optional but useful)
    print("\n===== GROQ RAW RESPONSE =====\n")
    print(raw[:1000])  # print first part only
    print("\n=============================\n")

    # --------------------------------------------------------
    # SAFE PARSE
    # --------------------------------------------------------
    data = _safe_load_llm_json(raw)

    tables = data.get("tables", [])
    columns = data.get("columns", [])

    if not isinstance(tables, list):
        tables = []

    if not isinstance(columns, list):
        columns = []

    return {
        "tables": tables,
        "columns": columns,
    }
