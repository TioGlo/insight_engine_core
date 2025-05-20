import os
from typing import List, Dict, Any, Optional
import ollama  # Ensure this is installed: uv pip install ollama

from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    DEFAULT_MODEL_NAME = "gemma3:12b-it-qat"  # A common Ollama default, or choose another
    DEFAULT_MAX_TOKENS = 512  # Ollama's num_predict, can be different from OpenAI's
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # Standard Ollama API endpoint

    def __init__(self,
                 model_name: Optional[str] = None,
                 host: Optional[str] = None,
                 timeout: Optional[int] = None,  # ollama.Client specific
                 **kwargs  # Other options for ollama.Client
                 ):
        """
        Initializes the Ollama LLM client.
        Args:
            model_name: The Ollama model to use (e.g., "llama3:latest", "mistral:latest").
                        Defaults to OllamaLLM.DEFAULT_MODEL_NAME.
            host: The host for the Ollama API. Defaults to http://localhost:11434.
            timeout: Optional timeout for client requests.
            **kwargs: Additional arguments for the ollama.Client.
        """
        super().__init__()
        self.model_name = model_name if model_name is not None else self.DEFAULT_MODEL_NAME
        self.host = host if host is not None else self.DEFAULT_OLLAMA_HOST

        client_params = {"host": self.host}
        if timeout is not None:
            client_params["timeout"] = timeout
        client_params.update(kwargs)  # Add any other client kwargs

        try:
            self.client = ollama.Client(**client_params)
            # You could optionally try a quick list models or heartbeat to verify connection
            # self.client.list()
            print(f"OllamaLLM initialized: model='{self.model_name}', host='{self.host}'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama client for host '{self.host}': {e}") from e

    def generate(self,
                 prompt: str,
                 context: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 max_tokens: Optional[int] = None,  # Will be mapped to num_predict
                 temperature: Optional[float] = None,
                 stop_sequences: Optional[List[str]] = None,  # Mapped to options.stop
                 system_message_content: Optional[str] = None,
                 stream: bool = False,  # Ollama supports streaming
                 format: Optional[str] = None,  # e.g., "json" if model supports
                 **kwargs  # For other Ollama options (e.g., top_p, top_k, seed)
                 ) -> str:
        """
        Generates a response from the Ollama LLM.
        """
        if not self.client:  # Should have been caught in __init__
            raise RuntimeError("Ollama client not initialized.")

        messages = []

        # 1. System Message (for context or general instructions)
        # Ollama typically uses the first message with role 'system' as the system prompt.
        # If both system_message_content and context are provided, combine them.
        effective_system_content = ""
        if system_message_content:
            effective_system_content += system_message_content

        if context:
            if effective_system_content:  # Append to existing system message
                effective_system_content += f"\n\nUse the following context to help answer the user's question:\n<context>\n{context}\n</context>"
            else:  # Create new system message for context
                effective_system_content = f"Use the following context to help answer the user's question:\n<context>\n{context}\n</context>"

        if effective_system_content:
            messages.append({"role": "system", "content": effective_system_content})

        # 2. Add history
        if history:
            messages.extend(history)

        # 3. Add the current user prompt
        messages.append({"role": "user", "content": prompt})

        # Prepare Ollama options
        options = kwargs  # Start with any passthrough kwargs
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif "num_predict" not in options:  # Apply default if not in kwargs
            options["num_predict"] = self.DEFAULT_MAX_TOKENS

        if temperature is not None:
            options["temperature"] = temperature
        elif "temperature" not in options:  # Apply default
            options["temperature"] = self.DEFAULT_TEMPERATURE

        if stop_sequences:  # Ollama expects 'stop' as a list of strings
            options["stop"] = stop_sequences

        # Other common Ollama options you might want to map from BaseLLM or add:
        # options["top_k"] = kwargs.get("top_k")
        # options["top_p"] = kwargs.get("top_p")
        # options["seed"] = kwargs.get("seed")
        # ... etc.

        try:
            print(f"OllamaLLM: Sending request to model '{self.model_name}' at host '{self.host}'")
            # print(f"OllamaLLM: Messages: {messages}") # Debug
            # print(f"OllamaLLM: Options: {options}")    # Debug

            if stream:  # Handle streaming if implemented (more complex)
                # response_stream = self.client.chat(
                #     model=self.model_name,
                #     messages=messages,
                #     stream=True,
                #     options=options,
                #     format=format
                # )
                # full_response_content = ""
                # for chunk in response_stream:
                #     if 'message' in chunk and 'content' in chunk['message']:
                #         full_response_content += chunk['message']['content']
                #     # Handle other parts of the stream if needed (e.g., done status)
                # return full_response_content.strip()
                raise NotImplementedError("Streaming not yet fully implemented in this example.")
            else:
                response_data = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options=options,
                    format=format  # Pass format if specified (e.g., "json")
                )

            response_content = response_data.get("message", {}).get("content")
            if response_content is None:
                print(f"OllamaLLM: Warning - Received None content from API. Full response: {response_data}")
                return ""
            return response_content.strip()

        except ollama.ResponseError as e:
            print(
                f"Ollama API Response Error (model: {self.model_name}, host: {self.host}): {e.status_code} - {e.error}")
            # You might want to re-raise a custom exception or return a specific error message
            raise RuntimeError(f"Ollama API call failed: {e.error} (Status: {e.status_code})") from e
        except Exception as e:  # Catch other errors like connection errors
            print(
                f"An unexpected error occurred during Ollama LLM generation (model: {self.model_name}, host: {self.host}): {e}")
            raise RuntimeError(f"Unexpected error in OllamaLLM: {e}") from e

    def get_model_name(self) -> str:
        return self.model_name


if __name__ == '__main__':
    print("Testing OllamaLLM...")
    # This test requires an Ollama server running locally with a model like 'llama3' or 'mistral' pulled.
    # Example: ollama pull llama3
    try:
        # Test with default model (ensure it's pulled in your Ollama server)
        print("\n--- Test 1: Default model (ensure Ollama server is running and model is pulled) ---")
        # You might need to change DEFAULT_MODEL_NAME if llama3 isn't your default/available
        llm = OllamaLLM(model_name="gemma3:12b-it-qat")  # Be explicit for testing

        response1 = llm.generate(prompt="What is the capital of France in a short sentence?", max_tokens=20)
        print(f"Response 1 ({llm.get_model_name()}): {response1}")

        print("\n--- Test 2: With context ---")
        context_str = "The Eiffel Tower is a famous landmark in Paris. Paris is known for its art and culture."
        response2 = llm.generate(
            prompt="What is Paris famous for, based on the context?",
            context=context_str,
            max_tokens=50,
            temperature=0.5
        )
        print(f"Response 2 ({llm.get_model_name()}): {response2}")

    except RuntimeError as e:
        print(f"Error during OllamaLLM direct test: {e}")
        print(
            "Ensure your Ollama server is running (e.g., 'ollama serve') and the model is pulled (e.g., 'ollama pull llama3').")
    except ImportError:
        print("Please install the ollama library: uv pip install ollama")