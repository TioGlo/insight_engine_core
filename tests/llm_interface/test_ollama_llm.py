import pytest
from unittest.mock import MagicMock, patch
from pytest_mock import MockerFixture

from insight_engine_core.llm_interface.ollama_llm import OllamaLLM  # Will fail: not created yet


# No specific API key needed from config for Ollama, but might need host/port if not default

# You might want a marker to skip these if Ollama library isn't installed
# or if a local Ollama server isn't expected to be running for these unit tests.
# For unit tests, we mock the library, so server isn't strictly needed.

# tests/llm_interface/test_ollama_llm.py
# ...
class TestOllamaLLM:
    @patch('insight_engine_core.llm_interface.ollama_llm.ollama.Client')  # Patch Client where it's used
    def test_ollama_llm_generate_simple_prompt(self, mock_ollama_client_constructor, mocker: MockerFixture):
        # 1. Configure the mock ollama.Client instance that the constructor will return
        mock_client_instance = MagicMock(name="MockOllamaClientInstance")

        # 2. Configure the 'chat' method ON THIS MOCK INSTANCE
        mock_ollama_response = {
            "message": {"content": "This is an Ollama test response."}
        }
        mock_client_instance.chat.return_value = mock_ollama_response  # client.chat() will return this

        # 3. Make the mocked constructor return our mock_client_instance
        mock_ollama_client_constructor.return_value = mock_client_instance

        # --- Instantiate OllamaLLM ---
        # Its __init__ will call ollama.Client(), which is now our mock_ollama_client_constructor,
        # which returns mock_client_instance. So, llm.client will be mock_client_instance.
        llm = OllamaLLM(model_name="test-model:latest", host="http://testhost:11434")

        prompt = "Hello, Ollama!"
        max_tokens = 70
        temperature = 0.6

        response_text = llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # --- Assertions ---
        # Assert ollama.Client was constructed (e.g. with host)
        mock_ollama_client_constructor.assert_called_once_with(host="http://testhost:11434")

        # Assert the 'chat' method on our mock_client_instance was called
        expected_messages = [{"role": "user", "content": prompt}]
        expected_options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        mock_client_instance.chat.assert_called_once_with(
            model="test-model:latest",
            messages=expected_messages,
            stream=False,
            options=expected_options,
            format=None  # Assuming format is None by default in generate
        )

        assert response_text == "This is an Ollama test response."
        assert llm.get_model_name() == "test-model:latest"

    # Apply similar patching to test_ollama_llm_generate_with_context_and_history
    @patch('insight_engine_core.llm_interface.ollama_llm.ollama.Client')
    def test_ollama_llm_generate_with_context_and_history(self, mock_ollama_client_constructor, mocker: MockerFixture):
        mock_client_instance = MagicMock(name="MockOllamaClientInstanceCtx")
        mock_ollama_response = {"message": {"content": "Ollama contextual response."}}
        mock_client_instance.chat.return_value = mock_ollama_response
        mock_ollama_client_constructor.return_value = mock_client_instance

        llm = OllamaLLM(model_name="context-model:latest")  # Uses default host

        # ... (rest of prompt, context, history setup) ...
        prompt = "What is the capital of France?"
        context = "Relevant context: Paris is a beautiful city."
        history = [
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is in Europe."}
        ]

        response_text = llm.generate(
            prompt=prompt,
            context=context,
            history=history
        )

        # ... (construct expected_messages and expected_options as before) ...
        expected_messages = [
            {"role": "system",
             "content": f"Use the following context to help answer the user's question:\n<context>\n{context}\n</context>"},
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is in Europe."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        expected_options = {
            "num_predict": OllamaLLM.DEFAULT_MAX_TOKENS,
            "temperature": OllamaLLM.DEFAULT_TEMPERATURE,
        }

        mock_client_instance.chat.assert_called_once_with(
            model="context-model:latest",
            messages=expected_messages,
            stream=False,
            options=expected_options,
            format=None
        )
        assert response_text == "Ollama contextual response."

    # test_ollama_llm_constructor_custom_host also needs to patch 'ollama.Client'
    # where it's imported in ollama_llm.py
    @patch('insight_engine_core.llm_interface.ollama_llm.ollama.Client')
    def test_ollama_llm_constructor_custom_host(self, mock_client_constructor, mocker: MockerFixture):
        mock_client_instance = MagicMock()  # Not strictly needed to configure its methods for this test
        mock_client_constructor.return_value = mock_client_instance

        custom_host = "http://customhost:12345"
        llm = OllamaLLM(model_name="llama3:latest", host=custom_host)

        mock_client_constructor.assert_called_once_with(host=custom_host)  # Check host was passed
        assert llm.model_name == "llama3:latest"
        assert llm.client is mock_client_instance
