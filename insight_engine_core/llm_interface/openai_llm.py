import os
from typing import List, Dict, Any, Optional
import openai  # Ensure this is installed: uv pip install openai>=1.3.0

from .base_llm import BaseLLM  # Import the abstract base class
from insight_engine_core import config as core_config
# from ..config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY  # Import API key from global config

CONFIG_OPENAI_API_KEY = core_config.get_openai_api_key()

class OpenAILLM(BaseLLM):
    """
    Concrete implementation of BaseLLM for interacting with OpenAI's Chat Completions API.
    """
    DEFAULT_MODEL_NAME = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = 1024  # Increased default for more comprehensive answers
    DEFAULT_TEMPERATURE = 0.7

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 **kwargs  # To catch other OpenAI client params if needed
                 ):
        """
        Initializes the OpenAI LLM client.
        Args:
            api_key: OpenAI API key. If None, attempts to use OPENAI_API_KEY from environment/config.
            model_name: The specific OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
                        Defaults to OpenAILLM.DEFAULT_MODEL_NAME.
            **kwargs: Additional arguments to pass to the openai.OpenAI client constructor.
        Raises:
            ValueError: If no API key is found.
        """
        super().__init__()  # Call to parent's __init__ if BaseLLM had one (good practice)

        resolved_api_key = api_key if api_key is not None else CONFIG_OPENAI_API_KEY

        if not resolved_api_key:
            # Try os.getenv as a final fallback if config didn't load it for some reason
            resolved_api_key = os.getenv("OPENAI_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "OpenAI API key not provided and not found in environment variables "
                    "or insight_engine_core.config.OPENAI_API_KEY."
                )

        self.api_key = resolved_api_key
        self.model_name = model_name if model_name is not None else self.DEFAULT_MODEL_NAME

        try:
            # Initialize the OpenAI client
            # Pass through any extra kwargs to the OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key, **kwargs)
            print(f"OpenAILLM initialized with model: {self.model_name}")
        except Exception as e:
            # Catch potential errors during client initialization (e.g., invalid key format though SDK might not check here)
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def generate(self,
                 prompt: str,
                 context: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 stop_sequences: Optional[List[str]] = None,
                 system_message_content: Optional[str] = None,  # Allow custom system message
                 **kwargs  # For other model-specific parameters like 'top_p', 'frequency_penalty'
                 ) -> str:
        """
        Generates a response from the OpenAI LLM.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")  # Should be caught by __init__

        messages = []

        # 1. Add a custom system message if provided
        if system_message_content:
            messages.append({"role": "system", "content": system_message_content})

        # 2. Add context as a system message (if no other system message and context exists)
        #    Or, append context to the custom system message.
        if context:
            if not system_message_content:  # If no custom system message, create one for context
                messages.append({"role": "system",
                                 "content": f"Use the following context to help answer the user's question:\n<context>\n{context}\n</context>"})
            else:  # Append context to existing custom system message
                messages[0]["content"] += f"\n\nRelevant Context:\n<context>\n{context}\n</context>"

        # 3. Add history
        if history:
            messages.extend(history)

        # 4. Add the current user prompt
        messages.append({"role": "user", "content": prompt})

        # Use provided parameters or fall back to class defaults or OpenAI API defaults
        final_max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        final_temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

        try:
            print(f"OpenAILLM: Sending request to model {self.model_name} with {len(messages)} messages.")
            # print(f"OpenAILLM: Messages: {messages}") # For debugging message structure

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=final_max_tokens,
                temperature=final_temperature,
                stop=stop_sequences if stop_sequences else None,  # API expects None, not empty list
                **kwargs  # Pass through other OpenAI specific params
            )

            response_content = completion.choices[0].message.content
            if response_content is None:
                # Handle cases where content might be None (e.g., if finish_reason is 'content_filter')
                print("OpenAILLM: Warning - Received None content from API.")
                return ""
            return response_content.strip()

        except openai.APIError as e:
            # Handle API errors (e.g., rate limits, server errors from OpenAI)
            print(f"OpenAI API Error: {e}")
            # You might want to re-raise a custom exception or return a specific error message
            raise RuntimeError(f"OpenAI API call failed: {e}") from e
        except Exception as e:
            # Handle other unexpected errors
            print(f"An unexpected error occurred during OpenAI LLM generation: {e}")
            raise RuntimeError(f"Unexpected error in OpenAILLM: {e}") from e

    def get_model_name(self) -> str:
        return self.model_name


if __name__ == '__main__':
    # This basic test requires OPENAI_API_KEY to be set in the environment
    # and for the config.py to load it.
    print("Testing OpenAILLM...")
    CONFIG_OPENAI_API_KEY = core_config.get_openai_api_key()
    if not CONFIG_OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        print("Skipping OpenAILLM direct test: OPENAI_API_KEY not found.")
    else:
        try:
            # Test with default model
            print("\n--- Test 1: Default model ---")
            llm = OpenAILLM()
            response1 = llm.generate(prompt="What is the capital of France in a short sentence?")
            print(f"Response 1 ({llm.get_model_name()}): {response1}")

            # Test with context
            print("\n--- Test 2: With context ---")
            llm_context = OpenAILLM(model_name="gpt-4o-mini")  # Can specify model
            context_str = "The Eiffel Tower is a famous landmark in Paris."
            response2 = llm_context.generate(
                prompt="What is Paris famous for?",
                context=context_str,
                max_tokens=50
            )
            print(f"Response 2 ({llm_context.get_model_name()}): {response2}")

            # Test with history
            print("\n--- Test 3: With history ---")
            history_list = [
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "That's a nice color!"}
            ]
            response3 = llm.generate(
                prompt="What did I say my favorite color was?",
                history=history_list,
                max_tokens=30
            )
            print(f"Response 3 ({llm.get_model_name()}): {response3}")

        except Exception as e:
            print(f"Error during OpenAILLM test: {e}")
