from __future__ import annotations

from typing import Dict, Any, List, Optional
from collections import Counter
import math
import json
import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from schema_matching_toolkit.common.db_config import DBConfig


# -----------------------------
# Helpers
# -----------------------------
def _get_engine(cfg: DBConfig) -> Engine:
    return create_engine(cfg.sqlalchemy_url())


def _safe_json(v):
    """Convert values like datetime/date/Decimal into JSON safe."""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def _entropy(values: List[Any]) -> float:
    """Shannon entropy for top values distribution."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return round(ent, 6)


def _infer_kind(dtype: str) -> str:
    dt = (dtype or "").lower()
    if any(x in dt for x in ["int", "numeric", "decimal", "float", "double", "real"]):
        return "numeric"
    if any(x in dt for x in ["date", "time", "timestamp"]):
        return "datetime"
    if any(x in dt for x in ["bool"]):
        return "boolean"
    return "text"


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _run_scalar(conn, sql: str, params: Optional[dict] = None, default=0):
    """Run a scalar query safely. If it fails, rollback and return default."""
    try:
        return conn.execute(text(sql), params or {}).scalar()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return default


def _run_fetchall(conn, sql: str, params: Optional[dict] = None):
    """Run a fetchall query safely. If it fails, rollback and return empty list."""
    try:
        return conn.execute(text(sql), params or {}).fetchall()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return []


def _run_first_mapping(conn, sql: str, params: Optional[dict] = None):
    """Run query and return first row as mapping safely."""
    try:
        return conn.execute(text(sql), params or {}).mappings().first()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None


# -----------------------------
# Main Profiling
# -----------------------------
def profile_schema(
    cfg: DBConfig,
    schema_data: Optional[Dict[str, Any]] = None,
    sample_size: int = 500,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Profiles ALL tables and ALL columns from schema.
    If schema_data not provided, it will query information_schema.

    Output:
    {
      "db_type": "...",
      "schema_name": "...",
      "table_count": ...,
      "tables": [
        {
          "table_name": "...",
          "row_count": ...,
          "columns": [
            { "column": "...", "data_type": "...", ...metrics... }
          ]
        }
      ]
    }
    """

    # Ensure safe ints
    sample_size = max(1, _safe_int(sample_size, 500))
    top_k = max(1, _safe_int(top_k, 10))

    engine = _get_engine(cfg)
    schema_name = cfg.schema_name or "public"

    # If schema_data not passed -> extract tables/columns from information_schema
    if schema_data is None:
        schema_data = {"tables": []}

        with engine.connect() as conn:
            tables = _run_fetchall(
                conn,
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                ORDER BY table_name
                """,
                {"schema": schema_name},
            )

            for (tname,) in tables:
                cols = _run_fetchall(
                    conn,
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                    """,
                    {"schema": schema_name, "table": tname},
                )

                schema_data["tables"].append(
                    {
                        "table_name": tname,
                        "columns": [{"column_name": c[0], "data_type": c[1]} for c in cols],
                    }
                )

    result: Dict[str, Any] = {
        "db_type": cfg.db_type,
        "schema_name": schema_name,
        "table_count": len(schema_data.get("tables", [])),
        "tables": [],
    }

    with engine.connect() as conn:
        for table in schema_data.get("tables", []):
            table_name = table.get("table_name")
            if not table_name:
                continue

            # -------------------------
            # Row count (SAFE)
            # -------------------------
            row_count = _run_scalar(
                conn,
                f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"',
                default=0,
            )
            row_count = int(row_count or 0)

            table_profile: Dict[str, Any] = {
                "table_name": table_name,
                "row_count": row_count,
                "columns": [],
            }

            for col in table.get("columns", []):
                col_name = col.get("column_name")
                data_type = col.get("data_type", "")
                if not col_name:
                    continue

                kind = _infer_kind(data_type)

                # -------------------------
                # Base stats (SAFE)
                # -------------------------
                null_count = _run_scalar(
                    conn,
                    f'''
                    SELECT COUNT(*)
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{col_name}" IS NULL
                    ''',
                    default=0,
                )
                null_count = int(null_count or 0)

                not_null_count = row_count - null_count
                null_percent = round((null_count / row_count) * 100, 4) if row_count else 0.0

                distinct_count = _run_scalar(
                    conn,
                    f'''
                    SELECT COUNT(DISTINCT "{col_name}")
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                    ''',
                    default=0,
                )
                distinct_count = int(distinct_count or 0)

                distinct_percent = round((distinct_count / row_count) * 100, 4) if row_count else 0.0
                duplicate_count = max(0, not_null_count - distinct_count)

                # -------------------------
                # Top values (SAFE)
                # -------------------------
                top_rows = _run_fetchall(
                    conn,
                    f'''
                    SELECT "{col_name}" AS value, COUNT(*) AS count
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                    GROUP BY "{col_name}"
                    ORDER BY count DESC
                    LIMIT {top_k}
                    '''
                )

                top_values = [{"value": _safe_json(r[0]), "count": int(r[1])} for r in top_rows]

                # -------------------------
                # Sample values (SAFE)
                # -------------------------
                sample_rows = _run_fetchall(
                    conn,
                    f'''
                    SELECT "{col_name}"
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                    LIMIT {sample_size}
                    '''
                )

                sample_values = [_safe_json(r[0]) for r in sample_rows[:20]]
                sample_for_entropy = [_safe_json(r[0]) for r in sample_rows[:200]]

                # -------------------------
                # Column profile base
                # -------------------------
                col_profile: Dict[str, Any] = {
                    "column": col_name,
                    "data_type": data_type,
                    "kind": kind,
                    "row_count": row_count,
                    "not_null_count": not_null_count,
                    "null_count": null_count,
                    "null_percent": null_percent,
                    "distinct_count": distinct_count,
                    "distinct_percent": distinct_percent,
                    "duplicate_count": duplicate_count,
                    "top_values": top_values,
                    "sample_values": sample_values,
                    "entropy": _entropy(sample_for_entropy),
                }

                # -------------------------
                # NUMERIC METRICS (SAFE)
                # -------------------------
                if kind == "numeric":
                    numeric_stats = _run_first_mapping(
                        conn,
                        f'''
                        SELECT
                            MIN("{col_name}") AS min_val,
                            MAX("{col_name}") AS max_val,
                            AVG("{col_name}") AS avg_val,
                            SUM("{col_name}") AS sum_val,
                            STDDEV_POP("{col_name}") AS stddev_val,
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col_name}") AS p25,
                            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{col_name}") AS median,
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col_name}") AS p75
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        '''
                    )

                    zero_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" = 0
                        ''',
                        default=0,
                    )

                    negative_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" < 0
                        ''',
                        default=0,
                    )

                    if numeric_stats:
                        numeric_stats_dict = {
                            "min": _safe_json(numeric_stats.get("min_val")),
                            "max": _safe_json(numeric_stats.get("max_val")),
                            "avg": float(numeric_stats["avg_val"]) if numeric_stats.get("avg_val") is not None else None,
                            "sum": float(numeric_stats["sum_val"]) if numeric_stats.get("sum_val") is not None else None,
                            "stddev": float(numeric_stats["stddev_val"]) if numeric_stats.get("stddev_val") is not None else None,
                            "p25": float(numeric_stats["p25"]) if numeric_stats.get("p25") is not None else None,
                            "median": float(numeric_stats["median"]) if numeric_stats.get("median") is not None else None,
                            "p75": float(numeric_stats["p75"]) if numeric_stats.get("p75") is not None else None,
                            "iqr": (
                                float(numeric_stats["p75"] - numeric_stats["p25"])
                                if numeric_stats.get("p75") is not None and numeric_stats.get("p25") is not None
                                else None
                            ),
                            "range": (
                                float(numeric_stats["max_val"] - numeric_stats["min_val"])
                                if numeric_stats.get("max_val") is not None and numeric_stats.get("min_val") is not None
                                else None
                            ),
                            "zero_count": int(zero_count or 0),
                            "negative_count": int(negative_count or 0),
                        }
                    else:
                        numeric_stats_dict = {
                            "min": None,
                            "max": None,
                            "avg": None,
                            "sum": None,
                            "stddev": None,
                            "p25": None,
                            "median": None,
                            "p75": None,
                            "iqr": None,
                            "range": None,
                            "zero_count": int(zero_count or 0),
                            "negative_count": int(negative_count or 0),
                        }

                    col_profile["numeric_stats"] = numeric_stats_dict

                # -------------------------
                # DATE/TIME METRICS (SAFE)
                # -------------------------
                if kind == "datetime":
                    date_row = _run_first_mapping(
                        conn,
                        f'''
                        SELECT
                            MIN("{col_name}") AS min_date,
                            MAX("{col_name}") AS max_date
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        '''
                    ) or {}

                    min_date = date_row.get("min_date")
                    max_date = date_row.get("max_date")

                    range_days = None
                    try:
                        range_days = _run_scalar(
                            conn,
                            f'''
                            SELECT (MAX("{col_name}") - MIN("{col_name}"))::int AS range_days
                            FROM "{schema_name}"."{table_name}"
                            WHERE "{col_name}" IS NOT NULL
                            ''',
                            default=None,
                        )
                    except Exception:
                        range_days = None

                    weekday_distribution = {}
                    most_common_weekday = None
                    try:
                        weekday_rows = _run_fetchall(
                            conn,
                            f'''
                            SELECT EXTRACT(DOW FROM "{col_name}")::int AS dow, COUNT(*) AS count
                            FROM "{schema_name}"."{table_name}"
                            WHERE "{col_name}" IS NOT NULL
                            GROUP BY dow
                            ORDER BY count DESC
                            '''
                        )

                        weekday_map = {
                            0: "Sunday",
                            1: "Monday",
                            2: "Tuesday",
                            3: "Wednesday",
                            4: "Thursday",
                            5: "Friday",
                            6: "Saturday",
                        }

                        weekday_distribution = {weekday_map[int(r[0])]: int(r[1]) for r in weekday_rows}
                        if weekday_rows:
                            most_common_weekday = weekday_map[int(weekday_rows[0][0])]
                    except Exception:
                        pass

                    month_distribution = {}
                    most_common_month = None
                    try:
                        month_rows = _run_fetchall(
                            conn,
                            f'''
                            SELECT EXTRACT(MONTH FROM "{col_name}")::int AS month, COUNT(*) AS count
                            FROM "{schema_name}"."{table_name}"
                            WHERE "{col_name}" IS NOT NULL
                            GROUP BY month
                            ORDER BY count DESC
                            '''
                        )

                        month_distribution = {int(r[0]): int(r[1]) for r in month_rows}
                        if month_rows:
                            most_common_month = int(month_rows[0][0])
                    except Exception:
                        pass

                    weekend_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND EXTRACT(DOW FROM "{col_name}") IN (0, 6)
                        ''',
                        default=None,
                    )

                    col_profile["date_stats"] = {
                        "min_date": _safe_json(min_date),
                        "max_date": _safe_json(max_date),
                        "range_days": int(range_days) if range_days is not None else None,
                        "weekday_distribution": weekday_distribution,
                        "most_common_weekday": most_common_weekday,
                        "month_distribution": month_distribution,
                        "most_common_month": most_common_month,
                        "weekend_count": int(weekend_count) if weekend_count is not None else None,
                    }

                # -------------------------
                # TEXT METRICS (SAFE)
                # -------------------------
                if kind == "text":
                    len_stats = _run_first_mapping(
                        conn,
                        f'''
                        SELECT
                            MIN(LENGTH(CAST("{col_name}" AS TEXT))) AS min_len,
                            MAX(LENGTH(CAST("{col_name}" AS TEXT))) AS max_len,
                            AVG(LENGTH(CAST("{col_name}" AS TEXT))) AS avg_len
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                        '''
                    ) or {}

                    empty_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND TRIM(CAST("{col_name}" AS TEXT)) = ''
                        ''',
                        default=0,
                    )

                    digit_only = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND CAST("{col_name}" AS TEXT) ~ '^[0-9]+$'
                        ''',
                        default=0,
                    )

                    alpha_only = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND CAST("{col_name}" AS TEXT) ~ '^[A-Za-z]+$'
                        ''',
                        default=0,
                    )

                    alnum_only = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND CAST("{col_name}" AS TEXT) ~ '^[A-Za-z0-9]+$'
                        ''',
                        default=0,
                    )

                    email_like = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" IS NOT NULL
                          AND CAST("{col_name}" AS TEXT) ~* '^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{{2,}}$'
                        ''',
                        default=0,
                    )

                    col_profile["text_stats"] = {
                        "min_length": int(len_stats.get("min_len")) if len_stats.get("min_len") is not None else None,
                        "max_length": int(len_stats.get("max_len")) if len_stats.get("max_len") is not None else None,
                        "avg_length": float(len_stats.get("avg_len")) if len_stats.get("avg_len") is not None else None,
                        "empty_string_count": int(empty_count or 0),
                        "digit_only_count": int(digit_only or 0),
                        "alpha_only_count": int(alpha_only or 0),
                        "alnum_only_count": int(alnum_only or 0),
                        "email_like_count": int(email_like or 0),
                    }

                # -------------------------
                # BOOLEAN METRICS (SAFE)
                # -------------------------
                if kind == "boolean":
                    true_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" = TRUE
                        ''',
                        default=0,
                    )

                    false_count = _run_scalar(
                        conn,
                        f'''
                        SELECT COUNT(*)
                        FROM "{schema_name}"."{table_name}"
                        WHERE "{col_name}" = FALSE
                        ''',
                        default=0,
                    )

                    col_profile["boolean_stats"] = {
                        "true_count": int(true_count or 0),
                        "false_count": int(false_count or 0),
                    }

                table_profile["columns"].append(col_profile)

            result["tables"].append(table_profile)

    return result
