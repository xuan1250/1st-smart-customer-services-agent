import chromadb
from chromadb.utils import embedding_functions
import json

class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.OpenAIEmbeddingsFunction(
            api_key="your_openai_key"
        )
        self.collection = self.client.create_collection(
            name="customer_service_kb",
            embedding_function=self.embedding_fn,
        )

    def add_document(self, documents: List[dict]):
        """Add FAQ documents to knowledge base"""
        for i, doc in enumerate(documents):
            self.collection.add(
                documents=[doc['content']],
                metadatas=[{"category": doc['category'], "id": doc['id']}],
                ids=[f"doc_{i}"],
            )

    def search(self, query: str, n_results: int = 3):
        """Search knowledge base for relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results