from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class SemanticSearchEngine:
    def __init__(self, host="localhost", port=6333, collection_name="customer_support", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the search engine with necessary configurations.
        :param host: Qdrant host
        :param port: Qdrant port
        :param collection_name: Name of the Qdrant collection
        :param model_name: Sentence Transformer model name
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=5):
        """
        Perform a semantic search in Qdrant.
        :param query: Search query (text)
        :param top_k: Number of top results to return
        :return: List of matching documents
        """
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [
            {
                "Customer Issue": res.payload["customer_issue"],
                "Category": res.payload["category"],
                "Resolution Response": res.payload["resolution_response"]
            }
            for res in results
        ]


if __name__ == "__main__":
    engine = SemanticSearchEngine()
    query = "I can't log in to my account."
    search_results = engine.search(query)

    for idx, result in enumerate(search_results, start=1):
        print(f"Result {idx}: {result}\n")
