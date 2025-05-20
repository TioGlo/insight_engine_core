from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sql_func # For func.l2_distance, etc.
from ..database.models import TextChunk  # Assuming TextChunk model is defined


@dataclass
class SimilaritySearchResult:
    chunk_id: int
    chunk_text: str
    metadata: Dict[str, Any]
    processed_text_source_id: int
    distance: Optional[float] = None  # PGVector can return distance

# from ..processing.embedder import EMBEDDING_DIMENSION # Or get from config/model
# For now, assume dimension matches what's in TextChunk.embedding column

# We'll need the actual embedding dimension for PGVector operations if not inferred
# For now, let's assume the TextChunk model's Vector column has the dimension defined.

class VectorStore:
    def __init__(self, db_session: Session):
        """
        Initializes the VectorStore with a SQLAlchemy session.
        Args:
            db_session: An active SQLAlchemy session.
        """
        self.db = db_session

    def add_documents(self,
                      processed_text_source_ids: List[int],
                      chunks: List[str],
                      embeddings: List[np.ndarray],
                      metadatas: List[Dict[str, Any]] = None,
                      chunk_orders: List[int] = None) -> List[TextChunk]:
        """
        Adds text chunks and their embeddings to the database.
        Args:
            processed_text_source_ids: List of IDs from ProcessedText table, one for each chunk.
            chunks: List of text strings (the chunks).
            embeddings: List of numpy arrays (the embeddings for each chunk).
            metadatas: Optional list of metadata dictionaries for each chunk.
            chunk_orders: Optional list of order integers for each chunk.

        Returns:
            List of created TextChunk ORM objects.
        """
        if not (len(chunks) == len(embeddings) == len(processed_text_source_ids)):
            raise ValueError("Lists of chunks, embeddings, and processed_text_source_ids must have the same length.")
        if metadatas and len(metadatas) != len(chunks):
            raise ValueError("If provided, metadatas list must have the same length as chunks.")
        if chunk_orders and len(chunk_orders) != len(chunks):
            raise ValueError("If provided, chunk_orders list must have the same length as chunks.")

        db_text_chunks = []
        for i, text_chunk_str in enumerate(chunks):
            metadata = metadatas[i] if metadatas else {}
            order = chunk_orders[i] if chunk_orders else i  # Default order to index

            db_chunk = TextChunk(
                processed_text_source_id=processed_text_source_ids[i],
                chunk_text=text_chunk_str,
                embedding=embeddings[i],  # PGVector expects a list or NumPy array
                metadata_=metadata,
                chunk_order=order
            )
            db_text_chunks.append(db_chunk)

        if db_text_chunks:
            self.db.add_all(db_text_chunks)
            # Commit is typically handled by the calling code that manages the session lifecycle
            # self.db.commit()
            # For unit testing, we might want to flush to get IDs
            self.db.flush()
            # Refresh to get DB-generated values like created_at if needed by tests
            # for chunk in db_text_chunks:
            #    self.db.refresh(chunk)
        return db_text_chunks

    # We will add similarity_search method next
    def similarity_search(self,
                          query_embedding: np.ndarray,
                          top_k: int = 5,
                          filter_metadata: Optional[Dict[str, Any]] = None  # For future use
                          ) -> List[SimilaritySearchResult]:
        """
        Performs a similarity search against the stored text chunks.
        Args:
            query_embedding: NumPy array of the query embedding.
            top_k: Number of top similar results to return.
            filter_metadata: Optional dictionary to filter results by metadata. (Not implemented yet)
        Returns:
            List of SimilaritySearchResult objects.
        """
        if not isinstance(query_embedding, np.ndarray):
            # PGVector expects a list or ndarray. Ensure it's in a usable format.
            # Or convert to list: query_embedding.tolist()
            pass

        # PGVector distance operators:
        # <-> : L2 distance
        # <#> : negative inner product (for normalized vectors, maximizing inner product is minimizing this)
        # <=> : cosine distance (1 - cosine_similarity)
        # We'll use cosine distance as it's common for semantic similarity.

        # Build the query
        stmt = (
            select(
                TextChunk.id,
                TextChunk.chunk_text,
                TextChunk.metadata_,
                TextChunk.processed_text_source_id,
                TextChunk.embedding.cosine_distance(query_embedding).label("distance")  # Calculate distance
            )
            .order_by(TextChunk.embedding.cosine_distance(query_embedding))  # Order by distance
            .limit(top_k)
        )

        # TODO: Add metadata filtering if filter_metadata is provided
        # This would involve adding .where() clauses based on JSONB operators on TextChunk.metadata_

        results = self.db.execute(stmt).fetchall()  # fetchall() gives list of Row objects

        search_results = []
        for row in results:
            search_results.append(
                SimilaritySearchResult(
                    chunk_id=row.id,
                    chunk_text=row.chunk_text,
                    metadata=row.metadata_,
                    processed_text_source_id=row.processed_text_source_id,
                    distance=row.distance
                )
            )
        return search_results