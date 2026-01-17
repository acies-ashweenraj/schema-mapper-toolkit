from typing import Dict, Any
from sqlalchemy import text

from schema_mapper_toolkit.common.db_config import DBConfig
from schema_mapper_toolkit.common.schema_models import SchemaMetadata, SchemaProfile, TableProfile, ColumnProfile
from schema_mapper_toolkit.extract.sqlalchemy_connector import get_engine
from schema_mapper_toolkit.common.exceptions import ProfilingError


def profile_schema(cfg: DBConfig, schema: SchemaMetadata, sample_limit: int = 200) -> SchemaProfile:
    try:
        engine = get_engine(cfg)
        tables_out = []

        with engine.connect() as conn:
            for t in schema.tables:
                col_profiles = []
                for c in t.columns:
                    col = c.column
                    table = t.table_name

                    # null percent
                    q_null = text(f"""
                        SELECT
                          COUNT(*) AS total,
                          SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS nulls
                        FROM "{cfg.schema_name}"."{table}"
                    """)

                    res = conn.execute(q_null).mappings().first()
                    total = float(res["total"] or 0)
                    nulls = float(res["nulls"] or 0)
                    null_percent = (nulls / total) * 100 if total > 0 else 0.0

                    # distinct count
                    q_distinct = text(f"""
                        SELECT COUNT(DISTINCT "{col}") AS dcount
                        FROM "{cfg.schema_name}"."{table}"
                    """)
                    dres = conn.execute(q_distinct).mappings().first()
                    distinct_count = int(dres["dcount"] or 0)

                    # top values
                    q_top = text(f"""
                        SELECT "{col}" AS value, COUNT(*) AS freq
                        FROM "{cfg.schema_name}"."{table}"
                        GROUP BY "{col}"
                        ORDER BY freq DESC
                        LIMIT 5
                    """)
                    top_values = [dict(x) for x in conn.execute(q_top).mappings().all()]

                    col_profiles.append(
                        ColumnProfile(
                            column=col,
                            null_percent=round(null_percent, 2),
                            distinct_count=distinct_count,
                            top_values=top_values,
                        )
                    )

                tables_out.append(TableProfile(table_name=t.table_name, columns=col_profiles))

        return SchemaProfile(schema_name=schema.schema_name, tables=tables_out)

    except Exception as e:
        raise ProfilingError(f"Profiling failed: {e}")
