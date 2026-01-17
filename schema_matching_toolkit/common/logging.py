class SchemaMapperError(Exception):
    pass


class DBConnectionError(SchemaMapperError):
    pass


class SchemaExtractError(SchemaMapperError):
    pass


class ProfilingError(SchemaMapperError):
    pass


class QdrantError(SchemaMapperError):
    pass


class GroqError(SchemaMapperError):
    pass
