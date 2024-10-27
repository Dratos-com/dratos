#from qdrant_client import QdrantClient
#from qdrant_client.http import models
#from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class QdrantMemory:
    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host, port=port)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Create a collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            ),
        )

    def add(self, content: List[str], metadata: List[Dict[str, Any]]) -> None:
        """
        Add content and metadata to the Qdrant collection.

        Args:
            content (List[str]): List of content to add.
            metadata (List[Dict[str, Any]]): List of metadata corresponding to each content item.
        """
        if len(content) != len(metadata):
            raise ValueError("The number of content items must match the number of metadata items.")

        vectors = self.encoder.encode(content)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx, 
                    vector=vector.tolist(), 
                    payload={**meta}
                )
                for idx, (vector, meta) in enumerate(zip(vectors, metadata))
            ]
        )

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the Qdrant collection.

        Args:
            query (str): The search query.
            agent_id (str): The ID of the agent to search for.
            limit (int): Maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: List of search results, each containing id, score, and text.
        """
        query_vector = self.encoder.encode(query).tolist()

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [
            {"id": result.id, "score": result.score, "product": result.payload}
            for result in search_result
        ]

# Example usage
if __name__ == "__main__":
    memory = QdrantMemory("my_collection")

    # Add content with metadata
    content = [
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
        "Qdrant is a vector database"
    ]
    metadata = [
        {"source": "fable", "category": "animals"},
        {"source": "latin", "category": "placeholder"},
        {"source": "tech", "category": "database"}
    ]
    memory.add(content, metadata)

    # Search
    query = "vector database"
    results = memory.search(query)

    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}, Content: {result['text']}")
        print(f"Metadata: {result['payload']}")
        print()
