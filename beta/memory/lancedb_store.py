import lancedb
from typing import List
from beta.memory.memory import Memory

class LanceDBMemoryStore:
    def __init__(self, db_path: str = "~/.lancedb"):
        self.db = lancedb.connect(db_path)
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        if "memories" not in self.db.table_names():
            return self.db.create_table("memories", schema={
                "id": str,
                "content": str,
                "vector": list
            })
        return self.db["memories"]

    def add_memory(self, memory: Memory):
        self.table.add([{
            "id": memory.id,
            "content": memory.content,
            "vector": memory.vector
        }])

    def search_memories(self, query_vector: List[float], limit: int = 5) -> List[Memory]:
        results = self.table.search(query_vector).limit(limit).to_list()
        return [Memory(id=r["id"], content=r["content"], vector=r["vector"]) for r in results]

    def reload_memories(self):
        # This method should be called after switching branches or time traveling
        self.table = self._get_or_create_table()

    def get_all_memories(self) -> List[Memory]:
        results = self.table.to_pandas()
        return [Memory(id=r["id"], content=r["content"], vector=r["vector"]) for _, r in results.iterrows()]