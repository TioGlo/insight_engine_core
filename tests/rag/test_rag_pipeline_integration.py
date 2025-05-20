import pytest
import numpy as np
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from sqlalchemy.orm import Session

# Imports from your project
from insight_engine_core.config import DATABASE_URL, EMBEDDING_MODEL_NAME
from insight_engine_core.database.models import Base, DataSource, RawDataItem, ProcessedText, TextChunk  # All models
from insight_engine_core.database.db_utils import init_db as core_init_db  # To create tables
from insight_engine_core.database.vector_store import VectorStore, SimilaritySearchResult
from insight_engine_core.processing.text_processor import TextProcessor
from insight_engine_core.processing.embedder import Embedder
from insight_engine_core.rag.retriever import Retriever
from insight_engine_core.rag.generator import Generator
from insight_engine_core.llm_interface.base_llm import BaseLLM  # For mocking

# Use the same PostgreSQL fixture as for vector_store integration tests
# Ensure this fixture is defined in a conftest.py accessible here or copy it.
# For now, I'll assume pg_db_session is available (e.g., from a conftest.py)
# If not, copy the pg_db_session fixture from test_vector_store_integration.py here.

# Let's redefine a minimal pg_db_session here for clarity if not using conftest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

pytestmark = [
    pytest.mark.integration_rag_pipeline,  # New marker
    pytest.mark.skipif(not DATABASE_URL or not DATABASE_URL.startswith("postgresql"),
                       reason="PostgreSQL DATABASE_URL not configured or not PostgreSQL")
]


@pytest.fixture(scope="function")
def pg_rag_session() -> Session:  # Renamed to avoid conflict if in same file as other pg_db_session
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with engine.connect() as connection:  # Verify pgvector
        result = connection.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).scalar_one_or_none()
        if not result: pytest.fail("pgvector extension not found.")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


class TestRagPipelineIntegration:
    @pytest.fixture(scope="class")  # Embedder can be class-scoped if model loading is slow
    def embedder_instance(self) -> Embedder:
        # This will load the real sentence transformer model once per test class
        try:
            return Embedder(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            pytest.skip(f"Failed to load embedder model ({EMBEDDING_MODEL_NAME}), skipping RAG pipeline tests: {e}")
            return None  # Should not be reached if pytest.skip works

    def test_generator_produces_answer_from_ingested_data(self, pg_rag_session: Session, embedder_instance: Embedder,
                                                          mocker: MockerFixture):
        """
        Tests the end-to-end RAG flow: ingest, retrieve, generate (with mocked LLM).
        This test will fail initially as the full pipeline isn't orchestrated yet,
        or the Generator's prompt construction might need tuning.
        """
        if embedder_instance is None:  # In case fixture failed and skipped
            pytest.skip("Embedder instance not available.")

        # 1. Setup: Instantiate components
        text_processor = TextProcessor(chunk_size=250, chunk_overlap=50)  # Small chunks for test
        vector_store = VectorStore(db_session=pg_rag_session)
        retriever = Retriever(embedder=embedder_instance, vector_store=vector_store)

        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.generate.return_value = "The capital of Testland is Testopolis, known for its great testing facilities."
        mock_llm.get_model_name.return_value = "mocked-llm-for-rag"

        generator = Generator(retriever=retriever, llm_interface=mock_llm)

        # 2. Ingest Sample Data
        # Create DataSource and RawDataItem
        data_source = DataSource(name="rag_test_docs", source_type="manual_test")
        pg_rag_session.add(data_source)
        pg_rag_session.flush()

        raw_doc_content = {
            "title": "About Testland",
            "body": "Testland is a fictional country. Its capital is Testopolis. Testopolis is famous for its excellent testing facilities and rigorous quality assurance processes. Many developers visit Testopolis."
        }
        raw_item = RawDataItem(data_source_id=data_source.id, source_internal_id="doc1", raw_content=raw_doc_content)
        pg_rag_session.add(raw_item)
        pg_rag_session.flush()

        # Create ProcessedText
        # For simplicity, let's assume the body is the main content to process
        processed_text = ProcessedText(raw_data_item_id=raw_item.id, cleaned_text=raw_doc_content["body"],
                                       text_type="body")
        pg_rag_session.add(processed_text)
        pg_rag_session.commit()  # Commit to get processed_text.id

        # Chunk, Embed, and Add to VectorStore (mimicking an ingestion pipeline)
        chunks_text = text_processor.process_and_chunk(processed_text.cleaned_text)
        chunk_embeddings = embedder_instance.embed_batch(chunks_text)

        vector_store.add_documents(
            processed_text_source_ids=[processed_text.id] * len(chunks_text),
            chunks=chunks_text,
            embeddings=chunk_embeddings,
            metadatas=[{"doc_id": "doc1", "part": i} for i in range(len(chunks_text))],
            chunk_orders=list(range(len(chunks_text)))
        )
        pg_rag_session.commit()

        # 3. Query using the Generator
        query = "What is Testopolis known for?"

        final_answer = generator.generate_response(query_text=query, retriever_top_k=3)

        # 4. Assertions
        # A. Check that the LLM was called
        mock_llm.generate.assert_called_once()

        # B. Inspect the prompt passed to the LLM to ensure context was included
        call_args = mock_llm.generate.call_args
        assert call_args is not None
        prompt_to_llm = call_args.kwargs.get('prompt') or call_args.args[0]  # Get the prompt

        print(f"\nPrompt sent to LLM:\n{prompt_to_llm}\n")  # For debugging

        assert "Testopolis" in prompt_to_llm  # Context should contain this
        assert "testing facilities" in prompt_to_llm  # Context should contain this
        assert query in prompt_to_llm  # Original query should be part of the prompt

        # C. Check the final answer (which is from our mocked LLM)
        assert final_answer == "The capital of Testland is Testopolis, known for its great testing facilities."