from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from schema_matching_toolkit.common.db_config import DBConfig


def _get_engine(cfg: DBConfig) -> Engine:
    return create_engine(cfg.sqlalchemy_url())


def _is_numeric(dtype: str) -> bool:
    dtype = (dtype or "").lower()
    return any(x in dtype for x in ["int", "numeric", "decimal", "float", "double", "real"])


def _is_text(dtype: str) -> bool:
    dtype = (dtype or "").lower()
    return any(x in dtype for x in ["char", "text", "varchar"])


def _is_date(dtype: str) -> bool:
    dtype = (dtype or "").lower()
    return any(x in dtype for x in ["date", "timestamp", "time"])


def profile_schema(
    cfg: DBConfig,
    schema: Dict[str, Any],
    top_values_k: int = 5,
    sample_k: int = 5,
) -> Dict[str, Any]:
    """
    Input:
      - cfg: DBConfig
      - schema: output of extract_schema()

    Output:
    {
      "db_type": "...",
      "schema_name": "...",
      "tables": [
        {
          "table_name": "...",
          "row_count": 123,
          "columns": [
            {
              "column": "col_name",
              "data_type": "...",
              "null_count": ...,
              "null_percent": ...,
              "distinct_count": ...,
              "distinct_percent": ...,
              "top_values": [{"value":..., "count":...}],
              "sample_values": [...],
              "numeric_stats": {...} (if numeric),
              "text_stats": {...} (if text),
              "date_stats": {...} (if date),
            }
          ]
        }
      ]
    }
    """
    engine = _get_engine(cfg)
    schema_name = schema.get("schema_name") or cfg.schema_name or "public"

    output = {
        "db_type": cfg.db_type,
        "schema_name": schema_name,
        "tables": [],
    }

    with engine.connect() as conn:
        for t in schema.get("tables", []):
            table_name = t["table_name"]
            columns = t.get("columns", [])

            # row count
            row_count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"')
            ).scalar() or 0

            table_profile = {
                "table_name": table_name,
                "row_count": int(row_count),
                "columns": [],
            }

            for c in columns:
                col = c["column_name"]
                dtype = c["data_type"]

                # null count
                null_count = conn.execute(
                    text(
                        f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}" WHERE "{col}" IS NULL'
                    )
                ).scalar() or 0

                null_percent = (null_count / row_count * 100) if row_count > 0 else 0.0

                # distinct count
                distinct_count = conn.execute(
                    text(
                        f'SELECT COUNT(DISTINCT "{col}") FROM "{schema_name}"."{table_name}"'
                    )
                ).scalar() or 0

                distinct_percent = (
                    (distinct_count / row_count * 100) if row_count > 0 else 0.0
                )

                # top values
                top_values = []
                try:
                    rows = conn.execute(
                        text(
                            f'''
                            SELECT "{col}" AS value, COUNT(*) AS cnt
                            FROM "{schema_name}"."{table_name}"
                            WHERE "{col}" IS NOT NULL
                            GROUP BY "{col}"
                            ORDER BY cnt DESC
                            LIMIT :k
                            '''
                        ),
                        {"k": top_values_k},
                    ).fetchall()

                    top_values = [{"value": r[0], "count": int(r[1])} for r in rows]
                except Exception:
                    top_values = []

                # sample values
                sample_values = []
                try:
                    rows = conn.execute(
                        text(
                            f'''
                            SELECT "{col}"
                            FROM "{schema_name}"."{table_name}"
                            WHERE "{col}" IS NOT NULL
                            LIMIT :k
                            '''
                        ),
                        {"k": sample_k},
                    ).fetchall()
                    sample_values = [r[0] for r in rows]
                except Exception:
                    sample_values = []

                col_profile: Dict[str, Any] = {
                    "column": col,
                    "data_type": dtype,
                    "null_count": int(null_count),
                    "null_percent": round(float(null_percent), 4),
                    "distinct_count": int(distinct_count),
                    "distinct_percent": round(float(distinct_percent), 4),
                    "top_values": top_values,
                    "sample_values": sample_values,
                }

                # numeric stats
                if _is_numeric(dtype):
                    try:
                        stats = conn.execute(
                            text(
                                f'''
                                SELECT
                                    MIN("{col}") AS min,
                                    MAX("{col}") AS max,
                                    AVG("{col}") AS avg,
                                    SUM("{col}") AS sum,
                                    STDDEV_POP("{col}") AS stddev
                                FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" IS NOT NULL
                                '''
                            )
                        ).mappings().first()

                        zero_count = conn.execute(
                            text(
                                f'''
                                SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" = 0
                                '''
                            )
                        ).scalar() or 0

                        negative_count = conn.execute(
                            text(
                                f'''
                                SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" < 0
                                '''
                            )
                        ).scalar() or 0

                        col_profile["numeric_stats"] = {
                            "min": stats["min"],
                            "max": stats["max"],
                            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
                            "sum": float(stats["sum"]) if stats["sum"] is not None else None,
                            "stddev": float(stats["stddev"]) if stats["stddev"] is not None else None,
                            "zero_count": int(zero_count),
                            "negative_count": int(negative_count),
                        }
                    except Exception:
                        col_profile["numeric_stats"] = {}

                # text stats
                if _is_text(dtype):
                    try:
                        stats = conn.execute(
                            text(
                                f'''
                                SELECT
                                    MIN(LENGTH("{col}")) AS min_length,
                                    MAX(LENGTH("{col}")) AS max_length,
                                    AVG(LENGTH("{col}")) AS avg_length
                                FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" IS NOT NULL
                                '''
                            )
                        ).mappings().first()

                        empty_string_count = conn.execute(
                            text(
                                f'''
                                SELECT COUNT(*) FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" = ''
                                '''
                            )
                        ).scalar() or 0

                        col_profile["text_stats"] = {
                            "min_length": stats["min_length"],
                            "max_length": stats["max_length"],
                            "avg_length": float(stats["avg_length"]) if stats["avg_length"] is not None else None,
                            "empty_string_count": int(empty_string_count),
                        }
                    except Exception:
                        col_profile["text_stats"] = {}

                # date stats
                if _is_date(dtype):
                    try:
                        stats = conn.execute(
                            text(
                                f'''
                                SELECT
                                    MIN("{col}") AS min_date,
                                    MAX("{col}") AS max_date
                                FROM "{schema_name}"."{table_name}"
                                WHERE "{col}" IS NOT NULL
                                '''
                            )
                        ).mappings().first()

                        col_profile["date_stats"] = {
                            "min_date": stats["min_date"],
                            "max_date": stats["max_date"],
                        }
                    except Exception:
                        col_profile["date_stats"] = {}

                table_profile["columns"].append(col_profile)

            output["tables"].append(table_profile)

    return output
