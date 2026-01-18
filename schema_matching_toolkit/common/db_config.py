from dataclasses import dataclass
from typing import Optional


@dataclass
class DBConfig:
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    schema_name: Optional[str] = None
    sqlite_path: Optional[str] = None

    def sqlalchemy_url(self) -> str:
        t = (self.db_type or "").lower().strip()

        if t in ["postgres", "postgresql"]:
            return (
                f"postgresql+psycopg2://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        if t == "mysql":
            return (
                f"mysql+pymysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        if t in ["mssql", "sqlserver"]:
            driver = "ODBC Driver 17 for SQL Server"
            return (
                f"mssql+pyodbc://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
                f"?driver={driver.replace(' ', '+')}"
            )

        if t == "oracle":
            return (
                f"oracle+oracledb://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/?service_name={self.database}"
            )

        if t == "sqlite":
            path = self.sqlite_path or ":memory:"
            return f"sqlite:///{path}"

        raise ValueError(f"Unsupported db_type: {self.db_type}")


@dataclass
class QdrantConfig:
    """
    One unified QdrantConfig for all modules
    """
    host: str = "localhost"
    port: int = 6333

    # collection name (each module can use different one)
    collection_name: str = "master_columns"

    # vector settings (MiniLM = 384, MPNet = 768, etc.)
    vector_name: str = "dense_vector"
    vector_size: int = 384


@dataclass
class GroqConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"
