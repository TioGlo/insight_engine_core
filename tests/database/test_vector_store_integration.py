import pytest
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from insight_engine_core.config import DATABASE_URL  # Your actual PG database URL
from insight_engine_core.config import MODEL_EMBEDDING_DIM
from insight_engine_core.database.models import Base, TextChunk, ProcessedText, RawDataItem, DataSource
from insight_engine_core.database.vector_store import VectorStore
from typing import List, Dict, Any

# Mark all tests in this file as integration tests requiring PostgreSQL
pytestmark = [
    pytest.mark.integration_pg,
    pytest.mark.skipif(not DATABASE_URL or not DATABASE_URL.startswith("postgresql"),
                       reason="PostgreSQL DATABASE_URL not configured or not PostgreSQL")
]


@pytest.fixture(scope="function")
def pg_db_session() -> Session:
    """
    Fixture to provide a SQLAlchemy session to a real PostgreSQL database
    and handle table creation and cleanup for each test function.
    """
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set, skipping PostgreSQL integration test.")

    engine = create_engine(DATABASE_URL, echo=False)  # echo=True for debugging SQL

    # Drop and recreate tables for a clean state for each test
    # This is destructive, ensure your DATABASE_URL is for a TEST database
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Verify pgvector extension (optional, but good sanity check)
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).scalar_one_or_none()
        if not result:
            pytest.fail(
                "pgvector extension not found in the test database. Please run: CREATE EXTENSION IF NOT EXISTS vector;")
        # Verify embedding dimension matches TextChunk model (if possible to check column type)
        # For now, we assume MODEL_EMBEDDING_DIM in models.py is correct for the DB.

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)  # Clean up after test


@pytest.fixture
def sample_processed_text_pg(pg_db_session: Session) -> ProcessedText:
    """Creates a sample ProcessedText record in the PostgreSQL test database."""
    data_source = DataSource(name="pg_test_source", source_type="test_pg")
    pg_db_session.add(data_source)
    pg_db_session.flush()

    raw_item = RawDataItem(
        data_source_id=data_source.id,
        source_internal_id="pg_raw_789",
        raw_content={"data": "some pg raw data"}
    )
    pg_db_session.add(raw_item)
    pg_db_session.flush()

    processed_text = ProcessedText(
        raw_data_item_id=raw_item.id,
        cleaned_text="Full text for PostgreSQL integration test.",
        text_type="body_pg"
    )
    pg_db_session.add(processed_text)
    pg_db_session.commit()  # Commit to get ID and make it available
    return processed_text


class TestVectorStoreIntegrationPG:
    def test_add_single_document_to_postgres(self, pg_db_session: Session, sample_processed_text_pg: ProcessedText):
        """
        Tests adding a single document with its embedding to PostgreSQL/PGVector.
        This test will currently fail because VectorStore.add_documents might not be fully implemented
        or might not handle PGVector specifics correctly yet.
        """
        vector_store = VectorStore(pg_db_session)

        processed_text_id = sample_processed_text_pg.id
        chunk_text = "This is a test chunk for PGVector."
        # Use the actual embedding dimension from your model
        embedding_np = np.random.rand(MODEL_EMBEDDING_DIM).astype(np.float32)
        metadata = {"test_key": "test_value"}
        chunk_order = 0

        # --- This is the action we are testing ---
        created_chunks_orm = vector_store.add_documents(
            processed_text_source_ids=[processed_text_id],
            chunks=[chunk_text],
            embeddings=[embedding_np],
            metadatas=[metadata],
            chunk_orders=[chunk_order]
        )
        pg_db_session.commit()  # Persist to the database

        # --- Assertions (These will likely fail initially) ---
        assert len(created_chunks_orm) == 1
        persisted_chunk_orm = created_chunks_orm[0]
        assert persisted_chunk_orm.id is not None
        assert persisted_chunk_orm.chunk_text == chunk_text
        assert persisted_chunk_orm.processed_text_source_id == processed_text_id
        assert persisted_chunk_orm.metadata_ == metadata
        assert persisted_chunk_orm.chunk_order == chunk_order

        # Verify the embedding was stored and can be retrieved
        # PGVector stores np.ndarray as its own type, or a list-like string.
        # When retrieved via SQLAlchemy, it should come back as a np.ndarray if the type is set up correctly.
        retrieved_db_chunk = pg_db_session.query(TextChunk).filter_by(id=persisted_chunk_orm.id).one()

        assert isinstance(retrieved_db_chunk.embedding, np.ndarray), \
            f"Embedding type is {type(retrieved_db_chunk.embedding)}, expected np.ndarray"
        assert retrieved_db_chunk.embedding.shape == (MODEL_EMBEDDING_DIM,)
        assert np.allclose(retrieved_db_chunk.embedding, embedding_np, atol=1e-6), \
            "Retrieved embedding does not match the original"

        # Optional: A raw SQL query to check how PGVector stored it (for debugging)
        with pg_db_session.connection() as conn:
            result = conn.execute(
                text(f"SELECT embedding FROM text_chunks WHERE id = {persisted_chunk_orm.id}")).scalar_one()
            print(
                f"\nRaw embedding from DB for chunk {persisted_chunk_orm.id}: {type(result)} - {str(result)[:100]}...")
            # This should be a string like '[0.1,0.2,...]' or a binary format if using bytea with pgvector
            # If using pgvector's native 'vector' type, psql \d text_chunks will show it.


# ... (keep existing imports and fixtures: pytestmark, pg_db_session, sample_processed_text_pg) ...

# For similarity search, we might want a dataclass for results
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimilaritySearchResult:
    chunk_id: int
    chunk_text: str
    metadata: Dict[str, Any]
    processed_text_source_id: int
    distance: Optional[float] = None  # PGVector can return distance


# ... (keep existing TestVectorStoreIntegrationPG class and its add_documents test) ...

class TestVectorStoreIntegrationPG:
    # ... (test_add_single_document_to_postgres - keep this) ...

    def test_similarity_search_finds_relevant_chunks(self, pg_db_session: Session,
                                                     sample_processed_text_pg: ProcessedText):
        """
        Tests adding documents and then performing a similarity search.
        This test will fail initially as similarity_search is not implemented.
        """
        vector_store = VectorStore(pg_db_session)

        # 1. Add some documents with known embeddings
        # Ensure MODEL_EMBEDDING_DIM is consistent (e.g., 3 for this test if easier)
        # For real tests, use the actual MODEL_EMBEDDING_DIM
        # Let's assume MODEL_EMBEDDING_DIM = 3 for simplicity of example vectors
        # IMPORTANT: If your TextChunk.embedding is Vector(384), these test vectors MUST be 384-dim.
        # For this example, I'll use dim 3. Adjust if your MODEL_EMBEDDING_DIM is different.
        # To make this robust, let's use the actual MODEL_EMBEDDING_DIM
        dim = MODEL_EMBEDDING_DIM

        docs_to_add = [
            {"id": sample_processed_text_pg.id, "chunk": "The cat sat on the mat.",
             "emb": np.array([0.1, 0.2, 0.9] + [0.0] * (dim - 3)), "meta": {"topic": "animals"}},
            {"id": sample_processed_text_pg.id, "chunk": "A dog played in the park.",
             "emb": np.array([0.8, 0.1, 0.3] + [0.0] * (dim - 3)), "meta": {"topic": "animals"}},
            {"id": sample_processed_text_pg.id, "chunk": "The weather is sunny today.",
             "emb": np.array([0.2, 0.7, 0.2] + [0.0] * (dim - 3)), "meta": {"topic": "weather"}},
            {"id": sample_processed_text_pg.id, "chunk": "A feline was resting on a rug.",
             "emb": np.array([0.15, 0.25, 0.85] + [0.0] * (dim - 3)), "meta": {"topic": "animals_alt"}},
            # Similar to cat
        ]

        vector_store.add_documents(
            processed_text_source_ids=[d["id"] for d in docs_to_add],
            chunks=[d["chunk"] for d in docs_to_add],
            embeddings=[d["emb"].astype(np.float32) for d in docs_to_add],
            metadatas=[d["meta"] for d in docs_to_add]
        )
        pg_db_session.commit()

        # 2. Define a query embedding (intentionally close to "cat" and "feline")
        query_embedding = np.array([0.12, 0.22, 0.88] + [0.0] * (dim - 3)).astype(np.float32)

        # --- This is the action we are testing ---
        # The similarity_search method doesn't exist yet, so this will cause an AttributeError
        # or if it exists but is empty, the assertions will fail.
        search_results: List[SimilaritySearchResult] = vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=2
            # filter_metadata={"topic": "animals"} # Optional: test filtering later
        )

        # --- Assertions (These will fail until implemented) ---
        assert len(search_results) == 2, "Should return top_k results"

        # PGVector L2 distance: smaller is more similar
        # PGVector cosine distance: smaller is more similar (distance = 1 - cosine_similarity)
        # PGVector inner product: larger is more similar (if vectors are normalized)
        # Let's assume cosine distance for now (default for <-> operator)

        # Result 1 should be "The cat sat on the mat." or "A feline was resting on a rug."
        # Result 2 should be the other one of those two.
        # We need to check content, not assume order if distances are very close.

        result_texts = [res.chunk_text for res in search_results]

        assert "The cat sat on the mat." in result_texts
        assert "A feline was resting on a rug." in result_texts

        # Check if distances are populated and ordered (smaller distance = better match)
        assert search_results[0].distance is not None, "Distance should be populated"
        assert search_results[1].distance is not None
        assert search_results[0].distance <= search_results[
            1].distance, "Results should be ordered by similarity (distance)"

        # Check metadata and other fields
        for res in search_results:
            assert res.chunk_id is not None
            assert res.processed_text_source_id == sample_processed_text_pg.id
            if res.chunk_text == "The cat sat on the mat.":
                assert res.metadata == {"topic": "animals"}
            elif res.chunk_text == "A feline was resting on a rug.":
                assert res.metadata == {"topic": "animals_alt"}