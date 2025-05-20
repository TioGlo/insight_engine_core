# insight_engine_core/processing/embedder.py (Corrected Lazy Loading)
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from ..config import EMBEDDING_MODEL_NAME  # Assuming this is just the string name

_model_instance = None
_model_dimension = None
_model_load_error = None
print(f"embedder.py: Initial module load: _model_instance={_model_instance}, _model_load_error={_model_load_error}")


def _load_global_model_if_needed():
    global _model_instance, _model_dimension, _model_load_error
    print(
        f"embedder.py: _load_global_model_if_needed called. Current state: _model_instance is {'set' if _model_instance else 'None'}, _model_load_error is {'set' if _model_load_error else 'None'}")
    if _model_instance is None and _model_load_error is None:
        try:
            print(f"Embedder: Lazily loading global model: {EMBEDDING_MODEL_NAME}...")
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)  # This will hit our mock
            _model_dimension = _model_instance.get_sentence_embedding_dimension()
            print(f"Embedder: Global model '{EMBEDDING_MODEL_NAME}' loaded. Dimension: {_model_dimension}")
            print(
                f"embedder.py: After successful load attempt: _model_instance is {'set' if _model_instance else 'None'}, _model_load_error is {'set' if _model_load_error else 'None'}")
        except Exception as e:
            _model_load_error = e
            print(f"Embedder: Error loading global model '{EMBEDDING_MODEL_NAME}': {e}")
            print(
                f"embedder.py: After FAILED load attempt: _model_instance is {'set' if _model_instance else 'None'}, _model_load_error is now: {_model_load_error}")
    else:
        print("embedder.py: _load_global_model_if_needed: Skipping load, already attempted.")


class Embedder:
    def __init__(self, model_name: str = None):
        self.model = None
        self.dimension = None
        print(
            f"Embedder.__init__: Called. model_name='{model_name}'. Initial module state: _model_instance is {'set' if _model_instance else 'None'}, _model_load_error is {'set' if _model_load_error else 'None'}")

        if model_name:
            # ... (instance specific logic - ensure this also uses the mocked SentenceTransformer for its own calls) ...
            try:
                print(f"Embedder (instance): Loading specific model: {model_name}...")
                self.model = SentenceTransformer(model_name)  # This will be mocked in tests
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"Embedder (instance): Model '{model_name}' loaded. Dimension: {self.dimension}")
            except Exception as e:
                raise RuntimeError(f"Failed to load instance-specific embedding model '{model_name}': {e}")
        else:  # Use the global model
            _load_global_model_if_needed()
            print(
                f"Embedder.__init__ (global path): After _load_global_model_if_needed: _model_instance is {'set' if _model_instance else 'None'}, _model_load_error is {'set' if _model_load_error else 'None'}")
            if _model_instance:
                self.model = _model_instance
                self.dimension = _model_dimension
            elif _model_load_error:
                raise RuntimeError(
                    f"Global embedding model '{EMBEDDING_MODEL_NAME}' previously failed to load: {_model_load_error}")
            else:
                raise RuntimeError(
                    f"Global embedding model '{EMBEDDING_MODEL_NAME}' could not be loaded and no specific model provided (should not happen if _load_global_model_if_needed ran).")
        print(f"Embedder.__init__: Exiting. self.model is {'set' if self.model else 'None'}")

    def embed(self, text: str) -> Union[np.ndarray, None]:
        if not self.model:
            raise RuntimeError("Embedder model not initialized properly.")
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Embedder: Error during text embedding: {e}")
            return None  # Or re-raise

    def embed_batch(self, texts: List[str]) -> Union[List[np.ndarray], None]:
        if not self.model:
            raise RuntimeError("Embedder model not initialized properly.")
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            print(f"Embedder: Error during batch text embedding: {e}")
            return None  # Or re-raise

    def get_dimension(self) -> Union[int, None]:
        return self.dimension


# This function can be called by other parts of the system if they need the dimension
# AFTER an Embedder instance has ensured the model is loaded.
# Or, models.py can just use a configured value.
def get_embedding_dimension_for_model(model_name_to_check: str = EMBEDDING_MODEL_NAME) -> int:
    """
    Utility to get dimension. Loads model if not already loaded.
    Preferably, dimension is known from config or a dedicated model info service.
    """
    if model_name_to_check == EMBEDDING_MODEL_NAME:
        _load_global_model_if_needed()
        if _model_dimension is not None:
            return _model_dimension
        elif _model_load_error:
            raise RuntimeError(
                f"Cannot get dimension, global model '{EMBEDDING_MODEL_NAME}' failed to load: {_model_load_error}")
        else:
            raise RuntimeError(f"Cannot get dimension, global model '{EMBEDDING_MODEL_NAME}' not loaded.")

    else:  # For a non-default model, load it temporarily to get dimension
        try:
            temp_model = SentenceTransformer(model_name_to_check)
            return temp_model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Could not load model '{model_name_to_check}' to get dimension: {e}")


# REMOVE THE FOLLOWING LINE or ensure it's only for direct script execution:
# EMBEDDING_DIMENSION = get_global_embedding_dimension() # THIS WAS THE PROBLEM

if __name__ == '__main__':
    # Test the embedder
    try:
        print("\n--- Testing Default Global Model Instance ---")
        # This will trigger the lazy load if not already loaded
        default_embedder = Embedder()
        print(f"Default embedder dimension: {default_embedder.get_dimension()}")

        example_text = "This is a test sentence."
        embedding = default_embedder.embed(example_text)
        if embedding is not None:
            print(f"Embedding (first 5 dims): {embedding[:5]}")

        # Test getting dimension via utility function
        # dim_util = get_embedding_dimension_for_model() # This will use the already loaded global model
        # print(f"Dimension from utility: {dim_util}")

    except RuntimeError as e:
        print(f"Error during __main__ test: {e}")
