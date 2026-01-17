from schema_mapper_toolkit.common.schema_models import SchemaMetadata, RelationshipGraph, RelationshipEdge


def find_relationships(schema: SchemaMetadata) -> RelationshipGraph:
    """
    Simple heuristic relationship finder:
    - If column name ends with _id and another table has same name as prefix -> relation
    Example: dept_fk_ref -> dept table dept_pk_id
    """
    edges = []
    all_cols = []
    for t in schema.tables:
        for c in t.columns:
            all_cols.append((t.table_name, c.column))

    for t in schema.tables:
        for c in t.columns:
            cname = c.column.lower()

            if cname.endswith("_id") or cname.endswith("_ref") or cname.endswith("_fk"):
                for t2 in schema.tables:
                    if t2.table_name.lower() in cname:
                        # pick first PK-like column
                        pk = None
                        for cc in t2.columns:
                            if "pk" in cc.column.lower() or cc.column.lower().endswith("_id"):
                                pk = cc.column
                                break
                        if pk:
                            edges.append(
                                RelationshipEdge(
                                    from_table=t.table_name,
                                    from_column=c.column,
                                    to_table=t2.table_name,
                                    to_column=pk,
                                    reason="heuristic_fk_match",
                                )
                            )

    return RelationshipGraph(edges=edges)
