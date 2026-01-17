from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from schema_mapper_toolkit.common.db_config import DBConfig
from schema_mapper_toolkit.common.exceptions import DBConnectionError


def get_engine(cfg: DBConfig) -> Engine:
    try:
        url = cfg.sqlalchemy_url()
        return create_engine(url)
    except Exception as e:
        raise DBConnectionError(f"Failed to create engine: {e}")
