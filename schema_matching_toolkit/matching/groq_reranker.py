from typing import Dict, Any, List
import json
import re
from groq import Groq


def safe_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Groq output was not valid JSON")


def groq_rerank(
    groq_api_key: str,
    groq_model: str,
    source_text: str,
    candidates: List[Dict[str, Any]],
):
    client = Groq(api_key=groq_api_key)

    prompt = f"""
You are a schema matching expert.

SOURCE:
{source_text}

CANDIDATES:
{candidates}

Rules:
- Choose best_match only from candidates list
- Return ONLY valid JSON

Return exactly:
{{
  "best_match": "table.column",
  "confidence": 0.0
}}
"""

    resp = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content
    return safe_json_parse(raw)
