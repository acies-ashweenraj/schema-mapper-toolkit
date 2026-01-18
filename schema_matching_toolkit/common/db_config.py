from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus


@dataclass
class DBConfig:
    """
    Universal DB config for SQLAlchemy.

    Supported db_type:
      - postgres / postgresql
      - mysql
      - mssql / sqlserver
      - oracle
      - sqlite
    """

    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    schema_name: Optional[str] = None

    # sqlite only
    sqlite_path: Optional[str] = None

    # MSSQL only
    mssql_driver: str = "ODBC Driver 17 for SQL Server"

    def sqlalchemy_url(self) -> str:
        t = (self.db_type or "").lower().strip()

        if t in ["postgres", "postgresql"]:
            return (
                f"postgresql+psycopg2://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        if t in ["mysql"]:
            return (
                f"mysql+pymysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        if t in ["mssql", "sqlserver"]:
            # driver must be URL encoded
            driver = quote_plus(self.mssql_driver)
            return (
                f"mssql+pyodbc://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}?driver={driver}"
            )

        if t in ["oracle"]:
            return (
                f"oracle+oracledb://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/?service_name={self.database}"
            )

        if t in ["sqlite"]:
            path = self.sqlite_path or ":memory:"
            return f"sqlite:///{path}"

        raise ValueError(f"Unsupported db_type: {self.db_type}")
