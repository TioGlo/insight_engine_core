from typing import List, Dict, Any, Optional  # Ensure these are imported
# from ..processing.embedder import Embedder # For type hints
# from ..database.vector_store import VectorStore, SimilaritySearchResult # For type hints
import numpy as np  # If used for type hints like np.ndarray


class Retriever:
    def __init__(self, embedder: 'Embedder', vector_store: 'VectorStore'):  # Use forward refs if needed
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query_text: str, top_k: int = 5,
                 filter_metadata: Optional[Dict[str, Any]] = None
                 ) -> List['SimilaritySearchResult']:  # Use forward ref for SimilaritySearchResult

        query_embedding = self.embedder.embed(query_text)
        if query_embedding is None:  # Should ideally not happen if embedder is robust
            print("Warning: Retriever received None embedding for query.")
            return []

        # Ensure query_embedding is a NumPy array if VectorStore expects it
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata  # Pass it through
        )
        return results
