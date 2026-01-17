from sqlalchemy import inspect
from schema_mapper_toolkit.extract.sqlalchemy_connector import get_engine
from schema_mapper_toolkit.common.schema_models import SchemaMetadata, TableMeta, ColumnMeta
from schema_mapper_toolkit.common.db_config import DBConfig
from schema_mapper_toolkit.common.exceptions import SchemaExtractError


def extract_schema(cfg: DBConfig) -> SchemaMetadata:
    try:
        engine = get_engine(cfg)
        inspector = inspect(engine)

        tables = inspector.get_table_names(schema=cfg.schema_name)

        table_objs = []
        for t in tables:
            cols = inspector.get_columns(t, schema=cfg.schema_name)
            col_objs = []
            for c in cols:
                col_id = f"{t}.{c['name']}"
                col_objs.append(
                    ColumnMeta(
                        id=col_id,
                        table=t,
                        column=c["name"],
                        type=str(c.get("type", "")),
                        nullable=bool(c.get("nullable", True)),
                    )
                )
            table_objs.append(TableMeta(table_name=t, columns=col_objs))

        return SchemaMetadata(
            db_type=cfg.db_type,
            schema_name=cfg.schema_name,
            tables=table_objs,
        )

    except Exception as e:
        raise SchemaExtractError(f"Schema extraction failed: {e}")
