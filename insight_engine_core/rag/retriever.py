from typing import Dict, List, Any, Optional
from insight_engine_core.processing.embedder import Embedder
from insight_engine_core.database.vector_store import VectorStore, SimilaritySearchResult

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query_text: str,
                 top_k: int = 5,
                 filter_metadata: Optional[Dict[str, Any]] = None) -> List[SimilaritySearchResult]:
        query_embedding = self.embedder.embed(query_text)

        # Test similarity of items
        similarity_search = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        return similarity_search
