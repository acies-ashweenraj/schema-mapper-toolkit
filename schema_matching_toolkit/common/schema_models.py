from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ColumnInfo:
    id: str                 # "table.column"
    table: str
    column: str
    type: str
    nullable: Optional[bool] = None
    primary_key: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableInfo:
    table_name: str
    columns: List[ColumnInfo] = field(default_factory=list)


@dataclass
class SchemaInfo:
    db_type: str
    schema_name: str
    tables: List[TableInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_type": self.db_type,
            "schema_name": self.schema_name,
            "tables": [
                {
                    "table_name": t.table_name,
                    "columns": [
                        {
                            "id": c.id,
                            "table": c.table,
                            "column": c.column,
                            "type": c.type,
                            "nullable": c.nullable,
                            "primary_key": c.primary_key,
                            "raw": c.raw,
                        }
                        for c in t.columns
                    ],
                }
                for t in self.tables
            ],
        }
