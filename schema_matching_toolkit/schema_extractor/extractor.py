from typing import Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from schema_matching_toolkit.common.db_config import DBConfig


def _get_engine(cfg: DBConfig) -> Engine:
    return create_engine(cfg.sqlalchemy_url())


def extract_schema(cfg: DBConfig) -> Dict[str, Any]:
    """
    Extract schema from DB.

    Output:
    {
      "db_type": "...",
      "schema_name": "...",
      "tables": [
        {
          "table_name": "...",
          "columns": [
            {
              "column_name": "...",
              "data_type": "...",
              "is_nullable": true/false
            }
          ]
        }
      ],
      "table_count": 0
    }
    """
    engine = _get_engine(cfg)
    db_type = (cfg.db_type or "").lower().strip()

    # default schema
    schema = cfg.schema_name or ("main" if db_type == "sqlite" else "public")

    # -------------------------
    # SQLITE SUPPORT
    # -------------------------
    if db_type == "sqlite":
        tables: List[Dict[str, Any]] = []

        with engine.connect() as conn:
            table_rows = conn.execute(
                text("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                    AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
            ).fetchall()

            for (table_name,) in table_rows:
                col_rows = conn.execute(text(f"PRAGMA table_info('{table_name}')")).fetchall()

                columns = []
                for r in col_rows:
                    # PRAGMA table_info output:
                    # (cid, name, type, notnull, dflt_value, pk)
                    col_name = r[1]
                    col_type = r[2] or "UNKNOWN"
                    notnull = int(r[3]) if r[3] is not None else 0

                    columns.append(
                        {
                            "column_name": col_name,
                            "data_type": col_type,
                            "is_nullable": (notnull == 0),
                        }
                    )

                tables.append({"table_name": table_name, "columns": columns})

        return {
            "db_type": cfg.db_type,
            "schema_name": schema,
            "tables": tables,
            "table_count": len(tables),
        }

    # -------------------------
    # INFORMATION_SCHEMA SUPPORT
    # -------------------------
    query_tables = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        ORDER BY table_name
    """)

    query_columns = text("""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
    """)

    tables: List[Dict[str, Any]] = []

    with engine.connect() as conn:
        table_rows = conn.execute(query_tables, {"schema": schema}).fetchall()
        col_rows = conn.execute(query_columns, {"schema": schema}).fetchall()

    table_names = [r[0] for r in table_rows]

    cols_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for r in col_rows:
        tname = r[0]
        cname = r[1]
        dtype = r[2]
        nullable = str(r[3]).lower() in ["yes", "true", "1"]

        cols_by_table.setdefault(tname, []).append(
            {
                "column_name": cname,
                "data_type": str(dtype),
                "is_nullable": nullable,
            }
        )

    for t in table_names:
        tables.append(
            {
                "table_name": t,
                "columns": cols_by_table.get(t, []),
            }
        )

    return {
        "db_type": cfg.db_type,
        "schema_name": schema,
        "tables": tables,
        "table_count": len(tables),
    }
