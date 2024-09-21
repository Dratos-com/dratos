from datetime import datetime
from typing import List, Optional
import lancedb
import pyarrow as pa
from pydantic import BaseModel, Field

class Memory(BaseModel):
    id: str = Field(..., description="Unique identifier for the memory")
    content: str = Field(..., description="Content of the memory")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the memory")
    vector: Optional[List[float]] = Field(None, description="Vector representation of the memory")

    @classmethod
    def schema(cls) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("vector", pa.list_(pa.float32(), 1536))  # Assuming 1536-dimensional embeddings
        ])

class MemoryStore:
    def __init__(self, db_uri: str, table_name: str = "memories"):
        self.db = lancedb.connect(db_uri)
        self.table_name = table_name
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, schema=Memory.schema())

    def add_memory(self, memory: Memory):
        table = self.db.open_table(self.table_name)
        table.add([memory.dict()])

    def search_memories(self, query_vector: List[float], limit: int = 5) -> List[Memory]:
        table = self.db.open_table(self.table_name)
        results = table.search(query_vector).limit(limit).to_list()
        return [Memory(**result) for result in results]

    def get_all_memories(self) -> List[Memory]:
        table = self.db.open_table(self.table_name)
        results = table.to_pandas()
        return [Memory(**row) for _, row in results.iterrows()]