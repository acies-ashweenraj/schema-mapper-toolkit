from dataclasses import dataclass
from typing import Dict, Any
import json
import re

from groq import Groq


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class GroqConfig:
    api_key: str


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Groq sometimes returns markdown or extra text.
    This extracts JSON safely.
    """
    text = (text or "").strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # extract JSON object {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Groq output was not valid JSON")


def describe_schema_with_groq(schema_data: Dict[str, Any], groq_cfg: GroqConfig) -> Dict[str, Any]:
    """
    INPUT:
      schema_data = output from extract_schema()

    OUTPUT:
      {
        "schema_name": "...",
        "tables": {
            "table_name": "table description"
        },
        "columns": {
            "table.column": "column description"
        }
      }
    """

    client = Groq(api_key=groq_cfg.api_key)

    prompt = f"""
You are a database metadata expert.

Given this database schema JSON, generate:
1) Description for each table
2) Description for each column

Return ONLY valid JSON (no markdown, no extra text) in this exact format:

{{
  "tables": {{
    "table_name": "table description"
  }},
  "columns": {{
    "table.column": "column description"
  }}
}}

Schema JSON:
{json.dumps(schema_data)}
"""

    resp = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,   # ✅ FIXED MODEL
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content
    parsed = _safe_json_parse(raw)

    return {
        "schema_name": schema_data.get("schema_name"),
        "tables": parsed.get("tables", {}),
        "columns": parsed.get("columns", {}),
    }
