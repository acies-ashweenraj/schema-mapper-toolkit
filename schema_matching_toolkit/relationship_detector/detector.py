from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from schema_matching_toolkit.common.db_config import DBConfig


def _get_engine(cfg: DBConfig) -> Engine:
    return create_engine(cfg.sqlalchemy_url())


def _normalize_col(col: str) -> str:
    return (col or "").lower().strip()


def _infer_fk_candidates(schema_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns list of (table_name, column_name) that look like FK columns.
    Example: dept_fk_ref, dept_id, customer_id, emp_ref_key
    """
    candidates = []
    for t in schema_data.get("tables", []):
        table = t.get("table_name")
        for c in t.get("columns", []):
            col = c.get("column_name")
            if not table or not col:
                continue
            name = _normalize_col(col)

            if name.endswith("_id") or name.endswith("_fk") or "_fk_" in name or "fk" in name:
                candidates.append((table, col))
            elif name.endswith("_ref") or name.endswith("_ref_key") or name.endswith("_ref_cd"):
                candidates.append((table, col))

    return candidates


def _find_pk_candidates(schema_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Heuristic: detect primary key-like column per table.
    Example: dept_pk_id, emp_unq_ref, id
    Returns: {table_name: pk_column}
    """
    pk_map = {}

    for t in schema_data.get("tables", []):
        table = t.get("table_name")
        cols = t.get("columns", [])
        if not table:
            continue

        # strongest guesses first
        priority = ["id", f"{table}_id", "pk_id", "pkid", "pk", "unq", "key", "ref"]

        best_col = None
        best_score = -1

        for c in cols:
            col = c.get("column_name")
            if not col:
                continue
            n = _normalize_col(col)

            score = 0
            if n == "id":
                score = 100
            elif n.endswith("_pk_id") or n.endswith("_pkid"):
                score = 95
            elif "pk" in n:
                score = 80
            elif n.endswith("_id"):
                score = 70
            elif "unq" in n or "unique" in n:
                score = 60
            elif "ref" in n and "id" in n:
                score = 50

            if score > best_score:
                best_score = score
                best_col = col

        if best_col:
            pk_map[table] = best_col

    return pk_map


def _get_constraints_postgres(engine: Engine, schema: str) -> List[Dict[str, Any]]:
    """
    Reads FK constraints from PostgreSQL information_schema.
    Returns list of relationships.
    """
    sql = text(
        """
        SELECT
            tc.table_name AS fk_table,
            kcu.column_name AS fk_column,
            ccu.table_name AS pk_table,
            ccu.column_name AS pk_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = :schema
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema}).fetchall()

    rels = []
    for r in rows:
        rels.append(
            {
                "fk_table": r[0],
                "fk_column": r[1],
                "pk_table": r[2],
                "pk_column": r[3],
                "confidence": 1.0,
            }
        )
    return rels


def _heuristic_relationships(schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If constraints not available, infer relationships using naming patterns.
    """
    fk_candidates = _infer_fk_candidates(schema_data)
    pk_map = _find_pk_candidates(schema_data)

    rels = []

    # create reverse map: pk_column -> table
    pk_col_to_table = {v: k for k, v in pk_map.items()}

    for fk_table, fk_col in fk_candidates:
        fk_norm = _normalize_col(fk_col)

        # try to match to a PK table by name similarity
        best_match: Optional[Tuple[str, str]] = None
        best_score = 0.0

        for pk_table, pk_col in pk_map.items():
            pk_norm = _normalize_col(pk_col)

            # very strong if fk contains pk table keyword
            if pk_table.lower().replace("tbl_", "") in fk_norm:
                score = 0.85
            # strong if fk contains pk col base
            elif pk_norm.replace("_pk_id", "").replace("_id", "") in fk_norm:
                score = 0.75
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_match = (pk_table, pk_col)

        if best_match and best_score > 0:
            pk_table, pk_col = best_match
            rels.append(
                {
                    "fk_table": fk_table,
                    "fk_column": fk_col,
                    "pk_table": pk_table,
                    "pk_column": pk_col,
                    "confidence": round(best_score, 4)
                }
            )

    return rels


def detect_relationships(cfg: DBConfig, schema_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry:
    - tries to detect FK constraints from DB (best)
    - fallback to heuristic relationship detection
    - returns schema enriched with edges

    OUTPUT:
      schema_data + edges per table + relationship_count
    """
    schema_name = cfg.schema_name or schema_data.get("schema_name") or "public"
    engine = _get_engine(cfg)

    relationships: List[Dict[str, Any]] = []

    # 1) Try DB constraints (Postgres supported)
    if cfg.db_type.lower() in ["postgres", "postgresql"]:
        try:
            relationships = _get_constraints_postgres(engine, schema_name)
        except Exception:
            relationships = []

    # 2) fallback heuristics
    if not relationships:
        relationships = _heuristic_relationships(schema_data)

    # 3) attach edges to tables
    table_edges: Dict[str, List[Dict[str, Any]]] = {}
    for rel in relationships:
        table_edges.setdefault(rel["fk_table"], []).append(rel)

    enriched = {"schema_name": schema_name, "tables": []}

    for t in schema_data.get("tables", []):
        table_name = t.get("table_name")
        enriched_table = dict(t)
        enriched_table["edges"] = table_edges.get(table_name, [])
        enriched["tables"].append(enriched_table)

    enriched["relationship_count"] = len(relationships)
    # enriched["relationship_method"] = "db_constraint" if any(r["method"] == "db_constraint" for r in relationships) else "heuristic"

    return enriched
