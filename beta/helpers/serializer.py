from abc import ABC, abstractmethod
import pyarrow as pa
from .result import Result

class PersistenceBackend(ABC):
    @abstractmethod
    def upsert(self, table: pa.Table, new_table: pa.Table, key_column: str) -> Result[pa.Table, Exception]:
        pass

class DeltaCatBackend(PersistenceBackend):
    def upsert(self, table: pa.Table, new_table: pa.Table, key_column: str) -> Result[pa.Table, Exception]:
        try:
            # Implement DeltaCat-specific upsert logic here
            updated_table = pa.concat_tables([table, new_table])  # Simplified example
            return Result.Ok(updated_table)
        except Exception as e:
            return Result.Error(e)

class Serializer:
    def __init__(self, backend: PersistenceBackend):
        self.backend = backend

    def serialize(self, obj) -> Result[pa.Table, Exception]:
        try:
            # Existing serialization logic
            serialized_table = pa.table(obj.__dict__)  # Simplified example
            return Result.Ok(serialized_table)
        except Exception as e:
            return Result.Error(e)

    def upsert(self, table: pa.Table, new_table: pa.Table, key_column: str) -> Result[pa.Table, Exception]:
        return self.backend.upsert(table, new_table, key_column)