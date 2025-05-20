import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pytest_mock import MockerFixture
from unittest.mock import MagicMock, call
from insight_engine_core.database.models import Base, TextChunk, ProcessedText, RawDataItem, \
    DataSource  # Import all relevant models
from insight_engine_core.database.vector_store import VectorStore

# Use an in-memory SQLite database for fast unit testing of VectorStore logic
# We won't test PGVector specific features here, just the SQLAlchemy interaction.
# For PGVector features, we'd need an integration test with a real PostgreSQL DB.
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")  # "function" scope for a fresh DB for each test
def db_session() -> Session:
    engine = create_engine(TEST_SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)  # Create tables in the in-memory DB

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)  # Clean up tables after test


@pytest.fixture
def sample_processed_text(db_session: Session) -> ProcessedText:
    # Helper to create a dummy ProcessedText parent for TextChunks
    # First, create its parent RawDataItem and DataSource
    data_source = DataSource(name="test_source", source_type="test")
    db_session.add(data_source)
    db_session.flush()  # Get data_source.id

    raw_item = RawDataItem(
        data_source_id=data_source.id,
        source_internal_id="test_raw_123",
        raw_content={"text": "some raw content"}
    )
    db_session.add(raw_item)
    db_session.flush()  # Get raw_item.id

    processed_text = ProcessedText(
        raw_data_item_id=raw_item.id,
        cleaned_text="This is the full processed text.",
        text_type="body"
    )
    db_session.add(processed_text)
    db_session.commit()  # Commit to get processed_text.id and make it available
    return processed_text


# tests/database/test_vector_store.py
# ... (imports) ...

# No need for db_session fixture or sample_processed_text for this pure unit test approach
# if we mock out all DB interactions.

class TestVectorStoreAddDocumentsPureUnit:
    def test_add_single_document_logic(self, mocker: MockerFixture):
        mock_db_session = MagicMock(spec=Session)  # Mock the SQLAlchemy Session

        # Mock the TextChunk class/constructor
        # When TextChunk is called, it returns a MagicMock instance
        mock_text_chunk_class = mocker.patch('insight_engine_core.database.vector_store.TextChunk', autospec=True)

        vector_store = VectorStore(mock_db_session)

        processed_text_ids = [1]
        chunks = ["This is chunk 1."]
        embeddings = [np.array([0.1, 0.2, 0.3])]
        metadatas = [{"source_page": 1}]
        chunk_orders = [0]

        created_mock_chunks = vector_store.add_documents(
            processed_text_source_ids=processed_text_ids,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            chunk_orders=chunk_orders
        )

        # Assert TextChunk constructor was called correctly
        mock_text_chunk_class.assert_called_once_with(
            processed_text_source_id=1,
            chunk_text="This is chunk 1.",
            embedding=[0.1, 0.2, 0.3],  # .tolist() is done inside add_documents
            metadata_={"source_page": 1},
            chunk_order=0
        )

        # Assert db.add_all was called with the list containing the mock TextChunk instance
        # The instance returned by mock_text_chunk_class() is itself a mock
        mock_db_session.add_all.assert_called_once()
        # Check that the argument to add_all was a list containing the instance our mocked TextChunk returned
        assert mock_db_session.add_all.call_args[0][0][0] is mock_text_chunk_class.return_value

        # Assert the returned list contains the mock TextChunk instance
        assert len(created_mock_chunks) == 1
        assert created_mock_chunks[0] is mock_text_chunk_class.return_value

        # Assert flush was called
        mock_db_session.flush.assert_called_once()

    def test_add_multiple_documents_with_metadata_and_orders_logic(self, mocker: MockerFixture):
        mock_db_session = MagicMock(spec=Session)
        mock_text_chunk_instances = [MagicMock(name=f"MockChunkMeta_{i}") for i in range(2)]
        mock_text_chunk_class = mocker.patch(
            'insight_engine_core.database.vector_store.TextChunk',
            side_effect=mock_text_chunk_instances
        )

        vector_store = VectorStore(mock_db_session)

        processed_text_ids = [100, 200]
        chunks = ["Meta chunk 0", "Meta chunk 1"]
        embeddings = [np.array([0.1, 0.1]), np.array([0.2, 0.2])]
        metadatas = [{"id": "m0"}, {"id": "m1"}]
        chunk_orders = [5, 6]

        created_mock_chunks = vector_store.add_documents(
            processed_text_source_ids=processed_text_ids,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            chunk_orders=chunk_orders
        )

        expected_calls = [
            call(processed_text_source_id=100, chunk_text="Meta chunk 0", embedding=[0.1, 0.1], metadata_={"id": "m0"},
                 chunk_order=5),
            call(processed_text_source_id=200, chunk_text="Meta chunk 1", embedding=[0.2, 0.2], metadata_={"id": "m1"},
                 chunk_order=6)
        ]
        mock_text_chunk_class.assert_has_calls(expected_calls, any_order=False)
        assert mock_text_chunk_class.call_count == 2
        mock_db_session.add_all.assert_called_once_with(mock_text_chunk_instances)
        assert created_mock_chunks == mock_text_chunk_instances
        mock_db_session.flush.assert_called_once()

    def test_add_documents_input_length_mismatch(self, db_session: Session):
        vector_store = VectorStore(db_session)
        with pytest.raises(ValueError,
                           match="Lists of chunks, embeddings, and processed_text_source_ids must have the same length."):
            vector_store.add_documents(processed_text_source_ids=[1], chunks=["c1", "c2"], embeddings=[np.array([0.1])])

        with pytest.raises(ValueError, match="If provided, metadatas list must have the same length as chunks."):
            vector_store.add_documents(processed_text_source_ids=[1], chunks=["c1"], embeddings=[np.array([0.1])],
                                       metadatas=[{}, {}])

    def test_add_empty_documents_list(self, db_session: Session):
        vector_store = VectorStore(db_session)
        created_chunks = vector_store.add_documents([], [], [])
        assert len(created_chunks) == 0
        # Ensure no commit error or anything if list is empty
        db_session.commit()
