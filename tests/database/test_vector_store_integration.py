# insight_engine_core/tests/database/test_vector_store_integration.py

import pytest
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dataclasses import dataclass, field # Keep this for SimilaritySearchResult
from typing import List, Dict, Any, Optional # Keep this

# Import the config module itself
from insight_engine_core import config as core_config

# Import models and VectorStore
from insight_engine_core.database.models import Base, TextChunk, ProcessedText, RawDataItem, DataSource
from insight_engine_core.database.vector_store import VectorStore, SimilaritySearchResult # Assuming SimilaritySearchResult is defined in vector_store.py or imported there

# --- Dynamically get config values ---
# These will be evaluated when pytest loads this test file.
# Ensure your insight_engine_core/tests/conftest.py loads .env if needed for these.
DATABASE_URL_FROM_CONFIG = core_config.get_database_url()
print(f"\n----------- DATABASE_URL_FROM_CONFIG: ------------ \n\n{DATABASE_URL_FROM_CONFIG}\n\n")
MODEL_EMBEDDING_DIM_FROM_CONFIG = core_config.get_model_embedding_dim()


# Mark all tests in this file as integration tests requiring PostgreSQL
pytestmark = [
    pytest.mark.integration_pg,
    pytest.mark.skipif(not DATABASE_URL_FROM_CONFIG or not DATABASE_URL_FROM_CONFIG.startswith("postgresql"),
                       reason="PostgreSQL DATABASE_URL not configured via core_config.get_database_url() or not PostgreSQL")
]


@pytest.fixture(scope="function")
def pg_db_session() -> Session:
    """
    Fixture to provide a SQLAlchemy session to a real PostgreSQL database
    and handle table creation and cleanup for each test function.
    Uses DATABASE_URL obtained from core_config.
    """
    # DATABASE_URL_FROM_CONFIG is already checked by pytestmark, but good to be explicit
    DATABASE_URL_FROM_CONFIG = core_config.get_database_url()
    if not DATABASE_URL_FROM_CONFIG:
        pytest.skip("DATABASE_URL not set (via core_config), skipping PostgreSQL integration test.")

    engine = create_engine(DATABASE_URL_FROM_CONFIG, echo=True)  # echo=True for debugging SQL

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).scalar_one_or_none()
        if not result:
            pytest.fail(
                "pgvector extension not found in the test database. Please run: CREATE EXTENSION IF NOT EXISTS vector;")
        # Further check: Ensure the embedding dimension used in tests matches the one TextChunk expects
        # This is implicitly handled by using MODEL_EMBEDDING_DIM_FROM_CONFIG in tests.

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_processed_text_pg(pg_db_session: Session) -> ProcessedText:
    """Creates a sample ProcessedText record in the PostgreSQL test database."""
    data_source = DataSource(name="pg_test_source", source_type="test_pg")
    pg_db_session.add(data_source)
    pg_db_session.flush() # Get data_source.id

    raw_item = RawDataItem(
        data_source_id=data_source.id,
        source_internal_id="pg_raw_789",
        raw_content={"data": "some pg raw data"}
    )
    pg_db_session.add(raw_item)
    pg_db_session.flush() # Get raw_item.id

    processed_text = ProcessedText(
        raw_data_item_id=raw_item.id,
        cleaned_text="Full text for PostgreSQL integration test.",
        text_type="body_pg"
    )
    pg_db_session.add(processed_text)
    pg_db_session.commit()
    return processed_text


# Note: The SimilaritySearchResult dataclass was defined in your original snippet.
# If it's defined in insight_engine_core.database.vector_store, ensure it's imported from there.
# For this example, I'll assume it's either defined here or correctly imported.
# If it's in vector_store.py, the import `from ...vector_store import VectorStore, SimilaritySearchResult` is correct.


class TestVectorStoreIntegrationPG:
    def test_add_single_document_to_postgres(self, pg_db_session: Session, sample_processed_text_pg: ProcessedText):
        vector_store = VectorStore(pg_db_session)
        print(f"\n----------- DATABASE_URL_FROM_CONFIG: ------------ \n\n{DATABASE_URL_FROM_CONFIG}\n\n")
        processed_text_id = sample_processed_text_pg.id
        chunk_text = "This is a test chunk for PGVector."
        # Use the embedding dimension from config
        embedding_np = np.random.rand(MODEL_EMBEDDING_DIM_FROM_CONFIG).astype(np.float32)
        metadata = {"test_key": "test_value"}
        chunk_order = 0

        created_chunks_orm = vector_store.add_documents(
            processed_text_source_ids=[processed_text_id],
            chunks=[chunk_text],
            embeddings=[embedding_np],
            metadatas=[metadata],
            chunk_orders=[chunk_order]
        )
        pg_db_session.commit()

        assert len(created_chunks_orm) == 1
        persisted_chunk_orm = created_chunks_orm[0]
        assert persisted_chunk_orm.id is not None
        assert persisted_chunk_orm.chunk_text == chunk_text
        assert persisted_chunk_orm.processed_text_source_id == processed_text_id
        assert persisted_chunk_orm.metadata_ == metadata
        assert persisted_chunk_orm.chunk_order == chunk_order

        retrieved_db_chunk = pg_db_session.query(TextChunk).filter_by(id=persisted_chunk_orm.id).one()

        assert isinstance(retrieved_db_chunk.embedding, np.ndarray), \
            f"Embedding type is {type(retrieved_db_chunk.embedding)}, expected np.ndarray"
        assert retrieved_db_chunk.embedding.shape == (MODEL_EMBEDDING_DIM_FROM_CONFIG,)
        assert np.allclose(retrieved_db_chunk.embedding, embedding_np, atol=1e-6), \
            "Retrieved embedding does not match the original"

        # Optional: Raw SQL query for debugging
        with pg_db_session.connection() as conn:
            result = conn.execute(
                text(f"SELECT embedding FROM text_chunks WHERE id = {persisted_chunk_orm.id}")).scalar_one()
            print(
                f"\nRaw embedding from DB for chunk {persisted_chunk_orm.id}: {type(result)} - {str(result)[:100]}...")


    def test_similarity_search_finds_relevant_chunks(self, pg_db_session: Session,
                                                     sample_processed_text_pg: ProcessedText):
        vector_store = VectorStore(pg_db_session)
        dim = MODEL_EMBEDDING_DIM_FROM_CONFIG # Use dimension from config

        # Create example embeddings ensuring they match the configured dimension
        def create_padded_vector(base_vector: List[float], target_dim: int) -> np.ndarray:
            if len(base_vector) > target_dim:
                raise ValueError(f"Base vector length {len(base_vector)} exceeds target dimension {target_dim}")
            return np.array(base_vector + [0.0] * (target_dim - len(base_vector))).astype(np.float32)

        docs_to_add = [
            {"id": sample_processed_text_pg.id, "chunk": "The cat sat on the mat.",
             "emb_base": [0.1, 0.2, 0.9], "meta": {"topic": "animals"}},
            {"id": sample_processed_text_pg.id, "chunk": "A dog played in the park.",
             "emb_base": [0.8, 0.1, 0.3], "meta": {"topic": "animals"}},
            {"id": sample_processed_text_pg.id, "chunk": "The weather is sunny today.",
             "emb_base": [0.2, 0.7, 0.2], "meta": {"topic": "weather"}},
            {"id": sample_processed_text_pg.id, "chunk": "A feline was resting on a rug.", # Similar to cat
             "emb_base": [0.15, 0.25, 0.85], "meta": {"topic": "animals_alt"}},
        ]

        # Prepare documents with correctly dimensioned embeddings
        processed_ids = [d["id"] for d in docs_to_add]
        chunks = [d["chunk"] for d in docs_to_add]
        embeddings = [create_padded_vector(d["emb_base"], dim) for d in docs_to_add]
        metadatas = [d["meta"] for d in docs_to_add]
        # Assuming chunk_orders are 0, 1, 2, 3 for simplicity or not strictly needed for this test focus
        chunk_orders = list(range(len(docs_to_add)))


        vector_store.add_documents(
            processed_text_source_ids=processed_ids,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            chunk_orders=chunk_orders # Pass chunk_orders
        )
        pg_db_session.commit()

        query_embedding_base = [0.12, 0.22, 0.88] # Intentionally close to "cat" and "feline"
        query_embedding = create_padded_vector(query_embedding_base, dim)

        search_results: List[SimilaritySearchResult] = vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=2
        )

        assert len(search_results) == 2, "Should return top_k results"
        result_texts = [res.chunk_text for res in search_results]

        assert "The cat sat on the mat." in result_texts
        assert "A feline was resting on a rug." in result_texts

        assert search_results[0].distance is not None, "Distance should be populated"
        assert search_results[1].distance is not None
        assert search_results[0].distance <= search_results[1].distance, \
            "Results should be ordered by similarity (smaller distance is more similar for cosine/L2)"

        for res in search_results:
            assert res.chunk_id is not None
            assert res.processed_text_source_id == sample_processed_text_pg.id
            if res.chunk_text == "The cat sat on the mat.":
                assert res.metadata == {"topic": "animals"}
            elif res.chunk_text == "A feline was resting on a rug.":
                assert res.metadata == {"topic": "animals_alt"}
