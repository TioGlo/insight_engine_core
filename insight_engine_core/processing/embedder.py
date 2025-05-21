# insight_engine_core/processing/embedder.py
import logging
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Import the config module to access getter functions
from .. import config

logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles text embedding using a SentenceTransformer model.
    The model is lazy-loaded upon first use.
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: int = 32):
        """
        Initializes the Embedder.

        Args:
            model_name (Optional[str]): The name of the SentenceTransformer model to use.
                                        If None, uses the default model from config.
            batch_size (int): The batch size for embedding.
        """
        # Determine the model name: use provided or fallback to config default
        self._model_name: str = model_name or config.get_embedding_model_name()
        self.batch_size: int = batch_size

        self._model_instance: Optional[SentenceTransformer] = None
        self._model_dimension: Optional[int] = None
        self._model_load_error: Optional[Exception] = None

        logger.debug(f"Embedder initialized for model: {self._model_name}. Model not yet loaded.")

    def _load_model_if_needed(self):
        """
        Lazily loads the SentenceTransformer model if it hasn't been loaded yet
        or if a previous attempt failed.
        """
        if self._model_instance is None and self._model_load_error is None:
            logger.info(f"Embedder: Lazily loading model: {self._model_name}...")
            try:
                self._model_instance = SentenceTransformer(self._model_name)
                # Determine and store the dimension once the model is loaded
                dimension_from_model = self._model_instance.get_sentence_embedding_dimension()
                if dimension_from_model is None:
                    logger.error(f"Model {self._model_name} returned None for dimension. This is unexpected.")
                    # Fallback to configured dimension if model doesn't provide one (should not happen)
                    self._model_dimension = config.get_model_embedding_dim()
                    logger.warning(
                        f"Using configured dimension {self._model_dimension} as fallback for {self._model_name}.")
                else:
                    self._model_dimension = dimension_from_model

                logger.info(f"Embedder: Model '{self._model_name}' loaded. Dimension: {self._model_dimension}")

            except Exception as e:
                self._model_load_error = e
                logger.error(f"Embedder: Error loading model '{self._model_name}': {e}", exc_info=True)
                # If model loading fails, we might still want to know the expected dimension from config
                # This helps if other parts of the system (like DB schema) need the dimension
                # even if embedding itself will fail.
                try:
                    self._model_dimension = config.get_model_embedding_dim()
                    logger.info(
                        f"Embedder: Setting dimension to configured {self._model_dimension} despite model load failure for {self._model_name}.")
                except Exception as config_e:
                    logger.error(
                        f"Embedder: Could not even get configured dimension after model load failure: {config_e}")
                    self._model_dimension = None  # Truly unknown

    def embed(self, text: str) -> np.ndarray:
        """
        Embeds a single piece of text.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedding vector.

        Raises:
            RuntimeError: If the model could not be loaded or if embedding fails.
        """
        self._load_model_if_needed()
        if self._model_load_error:
            raise RuntimeError(f"Cannot embed: Model '{self._model_name}' failed to load.") from self._model_load_error
        if not self._model_instance:
            # This case should ideally be caught by _model_load_error, but as a safeguard:
            raise RuntimeError(
                f"Cannot embed: Model '{self._model_name}' is not loaded and no load error was recorded.")

        try:
            # convert_to_numpy=True is default for SentenceTransformer.encode
            embedding = self._model_instance.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedder: Error during text embedding with model '{self._model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Embedding failed for model '{self._model_name}'.") from e

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a batch of texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            np.ndarray: A 2D NumPy array where each row is an embedding.

        Raises:
            RuntimeError: If the model could not be loaded or if embedding fails.
        """
        if not texts:
            return np.array([])  # Return empty array for empty list

        self._load_model_if_needed()
        if self._model_load_error:
            raise RuntimeError(
                f"Cannot embed batch: Model '{self._model_name}' failed to load.") from self._model_load_error
        if not self._model_instance:
            raise RuntimeError(
                f"Cannot embed batch: Model '{self._model_name}' is not loaded and no load error was recorded.")

        try:
            embeddings = self._model_instance.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False  # Typically off for library use
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedder: Error during batch text embedding with model '{self._model_name}': {e}",
                         exc_info=True)
            raise RuntimeError(f"Batch embedding failed for model '{self._model_name}'.") from e

    def get_dimension(self) -> int:
        """
        Returns the embedding dimension of the model.
        Loads the model if it hasn't been loaded yet.

        Returns:
            int: The embedding dimension.

        Raises:
            RuntimeError: If the dimension cannot be determined (e.g., model load failed and config is also unavailable).
        """
        self._load_model_if_needed()  # Ensures _model_dimension is set (or attempted)

        if self._model_dimension is not None:
            return self._model_dimension

        # If _model_dimension is still None here, it means loading failed AND fallback to config failed.
        # This is a critical state.
        if self._model_load_error:
            raise RuntimeError(
                f"Cannot determine dimension for model '{self._model_name}'. Model failed to load, and configured dimension also unavailable."
            ) from self._model_load_error
        else:
            # This state should ideally not be reached if _load_model_if_needed works correctly.
            raise RuntimeError(
                f"Cannot determine dimension for model '{self._model_name}'. Model not loaded and dimension unknown."
            )


# --- Main execution for simple testing ---
if __name__ == '__main__':
    # Configure basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # To test this, you might need a .env file in the insight_engine_core directory
    # or have EMBEDDING_MODEL_NAME set in your environment.
    # Example .env content:
    # EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

    print("\n--- Testing Embedder ---")
    try:
        # Test with default model from config
        print("\n1. Testing with default model (from config)...")
        default_embedder = Embedder()

        # First call to get_dimension will trigger model load
        dim = default_embedder.get_dimension()
        print(f"   Dimension (default model): {dim}")

        example_text = "This is a test sentence for the default embedder."
        embedding = default_embedder.embed(example_text)
        print(f"   Embedding shape (default model): {embedding.shape}")
        print(f"   Embedding (first 5 dims): {embedding[:5]}")

        example_batch = ["Batch sentence 1.", "Another sentence for the batch."]
        batch_embeddings = default_embedder.embed_batch(example_batch)
        print(f"   Batch embeddings shape (default model): {batch_embeddings.shape}")

        # Test with a specific model (if you have another small one, or it will re-use if same as default)
        # For this test, let's use the same as default to ensure it works.
        # If EMBEDDING_MODEL_NAME in .env is different, this will load another model.
        specific_model_name = config.get_embedding_model_name()  # Using the same for simplicity
        print(f"\n2. Testing with specific model: {specific_model_name}...")
        specific_embedder = Embedder(model_name=specific_model_name)

        dim_specific = specific_embedder.get_dimension()
        print(f"   Dimension (specific model): {dim_specific}")

        embedding_specific = specific_embedder.embed("Test sentence for specific model.")
        print(f"   Embedding shape (specific model): {embedding_specific.shape}")

        # Test scenario: Model fails to load (e.g., invalid model name)
        print("\n3. Testing with an invalid model name...")
        invalid_model_name = "this-model-does-not-exist-hopefully"
        try:
            error_embedder = Embedder(model_name=invalid_model_name)
            # The error will be raised when trying to use the embedder
            error_embedder.embed("This should fail.")
        except RuntimeError as e:
            print(f"   Successfully caught expected error for invalid model: {e}")
            # Check if get_dimension still provides configured fallback
            try:
                fallback_dim = error_embedder.get_dimension()  # Will try to get from config
                print(f"   Dimension from config after load error (invalid model): {fallback_dim}")
            except RuntimeError as e_dim:
                print(f"   Could not get dimension even from config after load error: {e_dim}")


    except Exception as e:
        logger.error(f"Error during __main__ test: {e}", exc_info=True)