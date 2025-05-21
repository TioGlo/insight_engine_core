# insight_engine_core/tests/processing/test_embedder.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the class to be tested
from insight_engine_core.processing.embedder import Embedder
# Import the config module to mock its functions
import insight_engine_core.config as core_config

# Default model name and dimension for consistent testing
TEST_MODEL_NAME = "test-model-name"
TEST_MODEL_DIMENSION = 128  # Example dimension for the test model


# --- Mocks for SentenceTransformer ---
@pytest.fixture
def mock_sentence_transformer_instance():
    """Mocks a SentenceTransformer model instance."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = TEST_MODEL_DIMENSION

    def mock_encode(texts, batch_size=None, show_progress_bar=None):  # Add other args if used
        if isinstance(texts, str):  # Single text
            return np.random.rand(TEST_MODEL_DIMENSION).astype(np.float32)
        # Batch of texts
        return np.random.rand(len(texts), TEST_MODEL_DIMENSION).astype(np.float32)

    mock_model.encode = MagicMock(side_effect=mock_encode)
    return mock_model


@pytest.fixture
def mock_sentence_transformer_class(mock_sentence_transformer_instance):
    """Mocks the SentenceTransformer class constructor."""
    mock_class = MagicMock(return_value=mock_sentence_transformer_instance)
    return mock_class


# --- Mocks for config module ---
@pytest.fixture
def mock_config_getters(mocker):
    """Mocks the config getter functions used by Embedder."""
    mocker.patch.object(core_config, 'get_embedding_model_name', return_value=TEST_MODEL_NAME)
    mocker.patch.object(core_config, 'get_model_embedding_dim', return_value=TEST_MODEL_DIMENSION)
    # Add mocks for other config getters if Embedder starts using them


# --- Main Test Class ---
@patch('insight_engine_core.processing.embedder.SentenceTransformer')  # Patch where it's LOOKED UP
def test_embedder_initialization_default_model(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test Embedder initialization with the default model name from (mocked) config."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance

    embedder = Embedder()

    # Check that the model name is set correctly from mocked config
    assert embedder._model_name == TEST_MODEL_NAME
    # Model should not be loaded yet
    assert embedder._model_instance is None
    assert embedder._model_dimension is None  # Dimension is set after load

    # Trigger model load by getting dimension
    dimension = embedder.get_dimension()
    assert dimension == TEST_MODEL_DIMENSION
    assert embedder._model_instance is mock_sentence_transformer_instance
    MockSentenceTransformerClass.assert_called_once_with(TEST_MODEL_NAME)


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_initialization_specific_model(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
        # mock_config_getters might not be strictly needed if specific model overrides default
):
    """Test Embedder initialization with a specific model name."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance
    specific_model = "specific-test-model"

    embedder = Embedder(model_name=specific_model)

    assert embedder._model_name == specific_model
    assert embedder._model_instance is None  # Not loaded yet

    # Trigger model load
    dimension = embedder.get_dimension()
    assert dimension == TEST_MODEL_DIMENSION
    assert embedder._model_instance is mock_sentence_transformer_instance
    MockSentenceTransformerClass.assert_called_once_with(specific_model)


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_get_dimension(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test get_dimension method, including lazy loading."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance
    embedder = Embedder()  # Uses default mocked model name

    # First call loads the model
    dimension = embedder.get_dimension()
    assert dimension == TEST_MODEL_DIMENSION
    assert embedder._model_instance is mock_sentence_transformer_instance
    MockSentenceTransformerClass.assert_called_once_with(TEST_MODEL_NAME)
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.assert_called_once()

    # Second call should use cached dimension and not reload
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.reset_mock()  # Reset for this check
    dimension_again = embedder.get_dimension()
    assert dimension_again == TEST_MODEL_DIMENSION
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.assert_not_called()  # Should not be called again


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_embed_single_text(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test embedding a single text."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance
    embedder = Embedder()
    test_text = "This is a test."

    embedding = embedder.embed(test_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (TEST_MODEL_DIMENSION,)
    mock_sentence_transformer_instance.encode.assert_called_once_with(test_text)


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_embed_batch_texts(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test embedding a batch of texts."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance
    embedder = Embedder()
    test_texts = ["Text 1.", "Text 2.", "Text 3."]

    embeddings = embedder.embed_batch(test_texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(test_texts), TEST_MODEL_DIMENSION)
    mock_sentence_transformer_instance.encode.assert_called_once_with(
        test_texts,
        batch_size=embedder.batch_size,  # or the default batch_size if not overridden
        show_progress_bar=False
    )


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_embed_empty_batch(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test embedding an empty batch."""
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance
    embedder = Embedder()

    embeddings = embedder.embed_batch([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0,)  # or (0, TEST_MODEL_DIMENSION) depending on np.array([]) behavior
    mock_sentence_transformer_instance.encode.assert_not_called()


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_model_load_failure(
        MockSentenceTransformerClass, mock_config_getters  # No mock_sentence_transformer_instance here
):
    """Test behavior when model loading fails."""
    load_error_message = "Simulated model load failure"
    MockSentenceTransformerClass.side_effect = ConnectionError(load_error_message)  # Simulate a typical error

    embedder = Embedder()  # Uses default mocked model name

    # Attempting to use the embedder should raise RuntimeError
    with pytest.raises(RuntimeError, match=f"Cannot embed: Model '{TEST_MODEL_NAME}' failed to load.") as excinfo:
        embedder.embed("This will fail.")
    assert isinstance(excinfo.value.__cause__, ConnectionError)  # Check the original cause
    assert load_error_message in str(excinfo.value.__cause__)

    # Check if get_dimension falls back to config after load error
    # The Embedder tries to set _model_dimension from config even on load failure
    dimension_after_fail = embedder.get_dimension()
    assert dimension_after_fail == TEST_MODEL_DIMENSION  # From mocked config.get_model_embedding_dim
    assert embedder._model_load_error is not None
    assert isinstance(embedder._model_load_error, ConnectionError)


@patch('insight_engine_core.processing.embedder.SentenceTransformer')
def test_embedder_model_get_dimension_returns_none(
        MockSentenceTransformerClass, mock_sentence_transformer_instance, mock_config_getters
):
    """Test when SentenceTransformer.get_sentence_embedding_dimension returns None."""
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.return_value = None  # Simulate problematic model
    MockSentenceTransformerClass.return_value = mock_sentence_transformer_instance

    embedder = Embedder()

    # get_dimension should fallback to config in this case
    dimension = embedder.get_dimension()
    assert dimension == TEST_MODEL_DIMENSION  # From mocked config.get_model_embedding_dim
    assert embedder._model_dimension == TEST_MODEL_DIMENSION


# Optional: Integration test (marked to be skipped by default or run selectively)
# This would not mock SentenceTransformer and would actually download a small model.
# It also wouldn't mock config, relying on a .env file or environment variables.
@pytest.mark.integration_embedder  # Custom marker, configure in pytest.ini
def test_embedder_integration_actual_model():
    """
    Integration test that actually loads a small model.
    Requires network access and sentence-transformers installed.
    Relies on EMBEDDING_MODEL_NAME being set in .env or environment
    to a small, fast-loading model like 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    # No mocking of SentenceTransformer or config here.
    # Ensure your test environment (e.g., .env file for insight_engine_core tests)
    # sets EMBEDDING_MODEL_NAME and MODEL_EMBEDDING_DIM appropriately.

    # For this test, let's explicitly use a known small model if config isn't set for it
    # Or, rely on the config.get_embedding_model_name() to provide a real one.
    model_name_for_integration = core_config.get_embedding_model_name()
    expected_dim_for_integration = core_config.get_model_embedding_dim()

    if "all-MiniLM-L6-v2" not in model_name_for_integration:  # Default to a known small one if config isn't specific
        model_name_for_integration = "sentence-transformers/all-MiniLM-L6-v2"
        expected_dim_for_integration = 384

    print(f"Running embedder integration test with model: {model_name_for_integration}")
    embedder = Embedder(model_name=model_name_for_integration)

    dimension = embedder.get_dimension()
    assert dimension == expected_dim_for_integration  # Check against known dimension for the model

    embedding = embedder.embed("Integration test sentence.")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (dimension,)

    batch_embeddings = embedder.embed_batch(["Batch text 1", "Batch text 2 for integration."])
    assert isinstance(batch_embeddings, np.ndarray)
    assert batch_embeddings.shape == (2, dimension)
    print(f"Embedder integration test with {model_name_for_integration} passed.")
