from typing import List, Optional
from .. import DataObject, Document as doc, Graph as kg


class KnowledgeBase(DataObject):
    artifacts = {}
    documents: Optional[List[document]]
    graphs = graph()

    async def add_object(self, obj: DataObject) -> str:
        self.objects[obj.id] = obj
        return obj.id

    async def get_object(self, object_id: str) -> Optional[DataObject]:
        return self.objects.get(object_id)

    async def search(self, query: str, limit: int) -> List[DataObject]:
        # Implement a simple search function
        # In a real-world scenario, you'd want to use a more sophisticated search algorithm
        results = []
        for obj in self.objects.values():
            if query.lower() in obj.content.lower():
                results.append(obj)
                if len(results) >= limit:
                    break
        return results
