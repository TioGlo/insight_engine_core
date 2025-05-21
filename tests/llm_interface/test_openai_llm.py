import pytest
from unittest.mock import MagicMock, patch
from pytest_mock import MockerFixture

from insight_engine_core.llm_interface.openai_llm import OpenAILLM  # Will fail: not created yet
from insight_engine_core import config as core_config

OPENAI_API_KEY = core_config.get_openai_api_key()

# Skip tests if OPENAI_API_KEY is not set in environment for actual calls (though we mock them)
# This is more for if you had integration tests later. For mocked tests, it's less critical.
pytestmark = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set in environment")


class TestOpenAILLM:
    @patch('openai.OpenAI')  # Patch the OpenAI client constructor where it's used in openai_llm.py
    def test_openai_llm_generate_simple_prompt(self, mock_openai_constructor, mocker: MockerFixture):
        """
        Tests basic generation with a simple prompt.
        This test will fail as OpenAILLM and its generate method are not implemented.
        """
        mock_chat_completion_instance = MagicMock()
        mock_chat_completion_instance.choices = [MagicMock(message=MagicMock(content="This is a test response."))]

        mock_openai_client_instance = MagicMock()
        mock_openai_client_instance.chat.completions.create.return_value = mock_chat_completion_instance

        mock_openai_constructor.return_value = mock_openai_client_instance  # OpenAI() returns our mock client

        # --- Instantiate the class to be tested ---
        llm = OpenAILLM(api_key="test_key_provided_to_constructor", model_name="gpt-3.5-turbo-test")

        prompt = "Hello, world!"
        max_tokens = 50
        temperature = 0.5

        # --- Call the method to be tested ---
        response_text = llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # --- Assertions ---
        # 1. Assert OpenAI client was initialized (implicitly by constructor not erroring)
        #    If api_key was passed to OpenAILLM constructor and then to OpenAI client:
        mock_openai_constructor.assert_called_once_with(api_key="test_key_provided_to_constructor")

        # 2. Assert chat.completions.create was called with correct parameters
        expected_messages = [{"role": "user", "content": prompt}]
        mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo-test",
            messages=expected_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None  # Default if not provided
        )

        # 3. Assert the response is correct
        assert response_text == "This is a test response."
        assert llm.get_model_name() == "gpt-3.5-turbo-test"

    @patch('openai.OpenAI')
    def test_openai_llm_generate_with_context_and_history(self, mock_openai_constructor, mocker: MockerFixture):
        mock_chat_completion_instance = MagicMock()
        mock_chat_completion_instance.choices = [MagicMock(message=MagicMock(content="Contextual response."))]

        mock_openai_client_instance = MagicMock()
        mock_openai_client_instance.chat.completions.create.return_value = mock_chat_completion_instance
        mock_openai_constructor.return_value = mock_openai_client_instance

        llm = OpenAILLM(model_name="gpt-4o-mini")  # Using API key from env for this one

        prompt = "What is the capital of France?"
        context = "Relevant context: Paris is a beautiful city known for the Eiffel Tower."
        history = [
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is a country in Europe."}
        ]

        response_text = llm.generate(
            prompt=prompt,
            context=context,
            history=history
        )

        # Construct expected messages for OpenAI API
        # System prompt for context, then history, then current user prompt
        expected_messages = [
            {"role": "system",
             "content": f"Use the following context to help answer the user's question:\n<context>\n{context}\n</context>"},
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is a country in Europe."},
            {"role": "user", "content": "What is the capital of France?"}
        ]

        mock_openai_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",  # This should match llm = OpenAILLM(model_name="gpt-4o-mini")
            messages=expected_messages,
            max_tokens=OpenAILLM.DEFAULT_MAX_TOKENS,  # Use the class default
            temperature=OpenAILLM.DEFAULT_TEMPERATURE,  # Use the class default
            stop=None
        )
        assert response_text == "Contextual response."

    def test_openai_llm_init_uses_env_api_key_if_not_provided(self, mocker: MockerFixture):
        # 1. Ensure that CONFIG_OPENAI_API_KEY (as imported in openai_llm.py) is None for this test.
        #    This forces the __init__ method to go past the check for CONFIG_OPENAI_API_KEY.
        mocker.patch('insight_engine_core.llm_interface.openai_llm.CONFIG_OPENAI_API_KEY', None)

        # 2. Mock os.getenv to control its return value when called for OPENAI_API_KEY.
        #    We need to patch it where it's called: in openai_llm.py (or globally if it's not imported there with 'as')
        #    The OpenAILLM class directly calls os.getenv.
        mock_os_getenv = mocker.patch('insight_engine_core.llm_interface.openai_llm.os.getenv')
        mock_os_getenv.return_value = "env_api_key_123"  # This is what os.getenv will return

        # 3. Mock the OpenAI client constructor
        mock_openai_constructor = mocker.patch('insight_engine_core.llm_interface.openai_llm.openai.OpenAI')
        # We don't need to set a return_value for mock_openai_constructor if we're just checking its call_args

        # --- Instantiate ---
        # api_key is not passed, CONFIG_OPENAI_API_KEY is mocked to None,
        # so it should fall back to os.getenv("OPENAI_API_KEY")
        llm = OpenAILLM()

        # --- Assertions ---
        # Check that os.getenv was indeed called with "OPENAI_API_KEY"
        mock_os_getenv.assert_any_call("OPENAI_API_KEY")
        # If os.getenv is called for other keys by other parts of init (e.g. by config.py itself during its import),
        # assert_any_call is safer. If it's ONLY called for OPENAI_API_KEY in this path,
        # assert_called_once_with("OPENAI_API_KEY") would also work.

        # Check that the OpenAI client was initialized with the key from our mocked os.getenv
        mock_openai_constructor.assert_called_once_with(api_key="env_api_key_123")