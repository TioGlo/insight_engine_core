# insight_engine_core/tests/conftest.py
import os
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)  # Or def pytest_configure(config):
def load_core_test_env():
    # __file__ is the path to this conftest.py file
    # os.path.dirname(__file__) is the 'tests' directory
    # os.path.join(os.path.dirname(__file__), '..') goes up one level to 'insight_engine_core/'
    # Then, '.env' is appended to get 'insight_engine_core/.env'
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

    # Use os.path.abspath to get the full path for clear debugging
    abs_env_path = os.path.abspath(env_path)
    print(f"DEBUG [insight_engine_core/tests/conftest.py]: Attempting to load .env from: {abs_env_path}")

    loaded = load_dotenv(dotenv_path=abs_env_path, override=True)

    if loaded:
        print(
            f"DEBUG [insight_engine_core/tests/conftest.py]: Successfully loaded .env. DATABASE_URL is now: {os.getenv('DATABASE_URL')}")
    else:
        print(
            f"DEBUG [insight_engine_core/tests/conftest.py]: .env file not found or not loaded from {abs_env_path}. Tests might use defaults or fail.")
        # You could add a pytest.fail here if the .env is critical for these tests
        # pytest.fail(f".env file not found at {abs_env_path}, which is required for core integration tests.")
