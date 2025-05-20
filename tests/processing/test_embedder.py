import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import importlib # Make sure this is imported
from pytest_mock import MockerFixture
import insight_engine_core.processing.embedder as embedder_module_to_test # Import at top
from insight_engine_core.config import EMBEDDING_MODEL_NAME # Get the name from config


def test_embedder_initialization_and_embed(mocker: MockerFixture):
    # 1. Define what our mock SentenceTransformer constructor should return for the GLOBAL model
    mock_global_model_instance = MagicMock(name="MockedST_GlobalInstance")
    mock_global_model_instance.get_sentence_embedding_dimension.return_value = 384
    mock_global_model_instance.encode.return_value = np.array([0.1, 0.2, 0.3])  # For global model's embed

    # 2. Reload the module
    importlib.reload(embedder_module_to_test)

    # 3. Patch SentenceTransformer on the reloaded module
    #    Initially, configure it to handle the global model load.
    #    We'll use a side_effect to change its behavior later for the specific model test.

    mock_specific_model_instance = MagicMock(name="MockedST_SpecificInstance")
    mock_specific_model_instance.get_sentence_embedding_dimension.return_value = 768  # Different dimension
    mock_specific_model_instance.encode.return_value = np.array([0.5, 0.6])  # Different embedding

    def constructor_side_effect(model_name_arg):
        print(f"TEST MOCK ST CONSTRUCTOR: Called with '{model_name_arg}'")
        if model_name_arg == EMBEDDING_MODEL_NAME:  # Default global model
            print(f"TEST MOCK ST CONSTRUCTOR: Returning global mock for '{model_name_arg}'")
            return mock_global_model_instance
        elif model_name_arg == "specific_model_test":
            print(f"TEST MOCK ST CONSTRUCTOR: Returning specific mock for '{model_name_arg}'")
            return mock_specific_model_instance
        else:
            # If called with an unexpected model name, raise an error or return a generic mock
            raise ValueError(f"Mock SentenceTransformer called with unexpected model name: {model_name_arg}")

    # Patch SentenceTransformer on the reloaded module using the side_effect
    # The mock object returned by mocker.patch.object IS the constructor mock itself
    the_mocked_constructor = mocker.patch.object(
        embedder_module_to_test,
        'SentenceTransformer',
        side_effect=constructor_side_effect
    )

    Embedder_reloaded = embedder_module_to_test.Embedder

    # --- Test Global Model Path ---
    print("TEST: Instantiating Embedder for global model...")
    embedder_global = Embedder_reloaded()

    the_mocked_constructor.assert_any_call(EMBEDDING_MODEL_NAME)  # Check it was called for global

    assert embedder_global.model is mock_global_model_instance, \
        f"Global embedder model is {embedder_global.model}, expected {mock_global_model_instance}"
    assert embedder_global.get_dimension() == 384

    text_global = "hello global"
    embedding_global = embedder_global.embed(text_global)
    mock_global_model_instance.encode.assert_called_once_with(text_global, convert_to_numpy=True)
    assert isinstance(embedding_global, np.ndarray)
    assert np.array_equal(embedding_global, np.array([0.1, 0.2, 0.3]))

    # --- Test Instance-Specific Model Path ---
    print("TEST: Instantiating Embedder for specific model...")
    # The side_effect on the_mocked_constructor is still active and will handle this new model_name
    embedder_specific = Embedder_reloaded(model_name="specific_model_test")

    the_mocked_constructor.assert_any_call("specific_model_test")  # Check it was called for specific

    assert embedder_specific.model is mock_specific_model_instance, \
        f"Specific embedder model is {embedder_specific.model}, expected {mock_specific_model_instance}"
    assert embedder_specific.get_dimension() == 768

    text_specific = "hello specific"
    embedding_specific = embedder_specific.embed(text_specific)
    mock_specific_model_instance.encode.assert_called_once_with(text_specific, convert_to_numpy=True)
    assert isinstance(embedding_specific, np.ndarray)
    assert np.array_equal(embedding_specific, np.array([0.5, 0.6]))

    # Check call counts on the constructor mock if needed
    # For example, to ensure it was called twice (once for global, once for specific)
    assert the_mocked_constructor.call_count == 2


def test_embedder_embed_batch(mocker: MockerFixture):  # Remove decorator, use mocker
    # 1. Define what our mock SentenceTransformer constructor should return
    mock_model_instance = MagicMock(name="MockedSTInstance_batch")
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384
    # Simulate batch encoding for this test
    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # 2. Reload the module to ensure a fresh state for its globals
    importlib.reload(embedder_module_to_test)

    # 3. Patch SentenceTransformer *directly on the reloaded module object*
    the_mocked_constructor = mocker.patch.object(
        embedder_module_to_test,
        'SentenceTransformer',
        return_value=mock_model_instance  # When ST() is called, it returns this mock_model_instance
    )

    Embedder_reloaded = embedder_module_to_test.Embedder

    # 4. Instantiate the Embedder. This should trigger _load_global_model_if_needed,
    #    which should now call our patched SentenceTransformer.
    embedder = Embedder_reloaded()

    # 5. Verify the patched constructor was called for the global model
    the_mocked_constructor.assert_called_once_with(EMBEDDING_MODEL_NAME)

    # 6. Verify the embedder instance is using our mock_model_instance
    assert embedder.model is mock_model_instance, \
        f"Embedder model is {embedder.model}, expected {mock_model_instance}"
    assert embedder.get_dimension() == 384

    texts = ["hello", "world"]
    embeddings = embedder.embed_batch(texts)

    # 7. Assert that the encode method on our mock_model_instance was called correctly
    mock_model_instance.encode.assert_called_once_with(texts,
                                                       convert_to_numpy=True, show_progress_bar=False)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 3)  # As per our mock's encode.return_value
    assert np.array_equal(embeddings[0], np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(embeddings[1], np.array([0.4, 0.5, 0.6]))


def test_embedder_global_model_load_fails(mocker: MockerFixture, capsys):
    # Reload the module to reset its global state (_model_instance, _model_load_error)
    importlib.reload(embedder_module_to_test)

    # Patch SentenceTransformer ON THE RELOADED MODULE to raise an error
    mock_st_constructor = mocker.patch.object(
        embedder_module_to_test,
        'SentenceTransformer',
        side_effect=Exception("Simulated global model load error")  # This makes ST() fail
    )

    Embedder_reloaded = embedder_module_to_test.Embedder

    # This call to Embedder_reloaded() should:
    # 1. Trigger _load_global_model_if_needed()
    # 2. _load_global_model_if_needed() calls SentenceTransformer() which is mocked to raise Exception
    # 3. The Exception should be caught, and _model_load_error should be set in embedder_module_to_test
    # 4. The Embedder constructor should then see _model_load_error is set and raise the RuntimeError
    with pytest.raises(RuntimeError, match="Global embedding model .* previously failed to load"):
        Embedder_reloaded()


def test_embedder_instance_specific_model_load_fails(mocker: MockerFixture, capsys):  # Type hint mocker
    # --- Setup for successful global model mock load ---
    mock_global_model_instance = MagicMock(name="GlobalMockInstance")
    mock_global_model_instance.get_sentence_embedding_dimension.return_value = 384

    # This will be the side effect for the SentenceTransformer constructor
    def constructor_side_effect_for_test(model_name_arg):
        print(f"Mock ST Constructor called with: {model_name_arg}")
        if model_name_arg == EMBEDDING_MODEL_NAME:
            print("Returning mock_global_model_instance")
            return mock_global_model_instance
        elif model_name_arg == "specific_fail_model":
            print("Raising error for specific_fail_model")
            raise Exception("Specific model load error")
        print(f"Unexpected model name '{model_name_arg}', returning new MagicMock")
        return MagicMock(name=f"UnexpectedMock_{model_name_arg}")

    # 1. Reload the module FIRST to get a fresh instance of it
    importlib.reload(embedder_module_to_test)

    # 2. NOW, patch SentenceTransformer *on the reloaded module object*
    #    The 'SentenceTransformer' name must exist as an import in embedder_module_to_test.
    mock_st_constructor = mocker.patch.object(
        embedder_module_to_test,
        'SentenceTransformer',  # The name 'SentenceTransformer' as imported in embedder.py
        side_effect=constructor_side_effect_for_test
    )

    Embedder_reloaded = embedder_module_to_test.Embedder

    # 3. Test global model loading path
    embedder_with_global = Embedder_reloaded()

    assert embedder_with_global.model is mock_global_model_instance, \
        f"Expected global model to be mock, got {embedder_with_global.model}"
    assert embedder_with_global.get_dimension() == 384
    mock_st_constructor.assert_any_call(EMBEDDING_MODEL_NAME)

    # --- Setup for instance-specific model load failure ---
    # The mock_st_constructor and its side_effect are still active.
    with pytest.raises(RuntimeError, match="Failed to load instance-specific embedding model 'specific_fail_model'"):
        Embedder_reloaded(model_name="specific_fail_model")

    mock_st_constructor.assert_any_call("specific_fail_model")


# Optional: An integration test (mark it to be skipped by default or run separately)
# This test would actually download and use the real model.
@pytest.mark.integration
def test_embedder_real_model_integration():
    pytest.importorskip("sentence_transformers")  # Skip if lib not installed
    pytest.importorskip("torch")
    from insight_engine_core.processing.embedder import Embedder, EMBEDDING_MODEL_NAME

    try:
        embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)  # Use the default configured model
        assert embedder.model is not None
        assert embedder.get_dimension() is not None

        text = "This is a real integration test."
        embedding = embedder.embed(text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_dimension(),)
    except Exception as e:
        pytest.fail(f"Embedder integration test failed: {e}")
