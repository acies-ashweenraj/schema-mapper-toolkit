from typing import Dict
from groq import Groq

from schema_mapper_toolkit.common.schema_models import SchemaMetadata, SchemaProfile
from schema_mapper_toolkit.common.exceptions import GroqError


def generate_descriptions_groq(
    schema: SchemaMetadata,
    profile: SchemaProfile,
    groq_api_key: str,
    groq_model: str = "llama-3.1-8b-instant",
) -> Dict[str, str]:
    """
    Input: schema + profile
    Output: {"table.column": "description"}
    """
    try:
        client = Groq(api_key=groq_api_key)

        descriptions = {}

        for table in schema.tables:
            for col in table.columns:
                col_id = col.id

                prompt = f"""
You are a data dictionary expert.

Column:
{col_id}
Type: {col.type}

Write a short business description in 1 sentence.
Return only the description text. No JSON. No markdown.
"""

                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )

                desc = resp.choices[0].message.content.strip()
                descriptions[col_id] = desc

        return descriptions

    except Exception as e:
        raise GroqError(f"Groq description generation failed: {e}")
