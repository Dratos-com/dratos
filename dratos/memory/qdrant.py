from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
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

    def add(self, documents: List[str], agent_id: str) -> None:
        """
        Add documents to the Qdrant collection.

        Args:
            documents (List[str]): List of documents to add.
            agent_id (str): The ID of the agent associated with these documents.
        """
        vectors = self.encoder.encode(documents)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=idx, vector=vector.tolist(), payload={"text": text, "agent_id": agent_id})
                for idx, (text, vector) in enumerate(zip(documents, vectors))
            ]
        )

    def search(self, query: str, agent_id: str, limit: int = 3) -> List[Dict[str, Any]]:
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
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="agent_id",
                        match=models.MatchValue(value=agent_id)
                    )
                ]
            ),
            limit=limit
        )

        return [
            {"id": result.id, "score": result.score, "text": result.payload['text']}
            for result in search_result
        ]

# Example usage
if __name__ == "__main__":
    memory = QdrantMemory("my_collection")

    # Add documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
        "Qdrant is a vector database"
    ]
    memory.add(documents, "agent_123")

    # Search
    query = "vector database"
    results = memory.search(query, "agent_123")

    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}, Text: {result['text']}")
