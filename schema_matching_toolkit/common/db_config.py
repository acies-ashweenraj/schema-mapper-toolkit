from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DBConfig:
    db_type: str                 # postgres | mysql | mssql | sqlite | oracle...
    host: str = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    schema_name: str = "public"

    def sqlalchemy_url(self) -> str:
        """
        Returns SQLAlchemy URL string for multiple DB types.
        """
        db = (self.db_type or "").lower().strip()

        # sqlite special case
        if db == "sqlite":
            # example: database="mydb.sqlite"
            if not self.database:
                raise ValueError("SQLite requires database file path in database field")
            return f"sqlite:///{self.database}"

        # default for network DBs
        if not self.database:
            raise ValueError("database is required")
        if not self.username:
            raise ValueError("username is required")
        if self.password is None:
            raise ValueError("password is required")
        if not self.host:
            raise ValueError("host is required")
        if not self.port:
            raise ValueError("port is required")

        # Mapping db_type -> driver
        # You can extend later
        if db == "postgres":
            driver = "postgresql+psycopg2"
        elif db == "mysql":
            driver = "mysql+pymysql"
        elif db in ["mssql", "sqlserver"]:
            driver = "mssql+pyodbc"
        elif db == "oracle":
            driver = "oracle+cx_oracle"
        else:
            # allow user to pass full SQLAlchemy dialect
            driver = db

        return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DBConfig":
        return cls(
            db_type=data.get("db_type", "postgres"),
            host=data.get("host", "localhost"),
            port=data.get("port"),
            database=data.get("database"),
            username=data.get("username"),
            password=data.get("password"),
            schema_name=data.get("schema_name", "public"),
        )
