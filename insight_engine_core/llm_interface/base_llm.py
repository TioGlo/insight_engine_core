from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    @abstractmethod
    def generate(self,
                 prompt: str,
                 context: Optional[str] = None, # For RAG context
                 history: Optional[List[Dict[str, str]]] = None, # For chat history
                 max_tokens: Optional[int] = 150,
                 temperature: Optional[float] = 0.7,
                 stop_sequences: Optional[List[str]] = None,
                 **kwargs # For other model-specific parameters
                ) -> str:
        """
        Generates a response from the LLM based on a prompt and optional context/history.
        Args:
            prompt: The main user prompt or question.
            context: Optional context string (e.g., retrieved documents for RAG).
            history: Optional list of previous turns in a conversation, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}].
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences at which to stop generation.
            **kwargs: Additional model-specific parameters.
        Returns:
            The LLM's generated text response.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the name of the LLM model being used."""
        pass
    