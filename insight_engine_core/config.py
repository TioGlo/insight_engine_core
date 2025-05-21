# insight_engine_core/config.py
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Add a print statement at the very top of this module to see when it's imported
print(f"DEBUG [insight_engine_core/config.py]: Module imported. Current DATABASE_URL from os.environ: {os.environ.get('DATABASE_URL')}")

def get_database_url():
    # Add a print inside the function to see when it's called and what os.getenv sees
    env_db_url = os.getenv("DATABASE_URL")
    print(f"DEBUG [insight_engine_core/config.py - get_database_url()]: Called. os.getenv('DATABASE_URL') returned: {env_db_url}")
    if not env_db_url: # Checks for None or empty string
        logger.warning("DATABASE_URL not set in environment. Using default for get_database_url().")
        return "postgresql://test_user:test_pass@localhost:5432/test_insight_engine_db" # Fallback
    return env_db_url

def get_embedding_model_name():
    env_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"DEBUG [insight_engine_core/config.py - get_embedding_model_name()]: Called. Returning: {env_model_name}")
    return env_model_name

def get_model_embedding_dim():
    model_name = get_embedding_model_name() # This will call the function above
    default_dim_for_model = 384 # Default
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        default_dim_for_model = 384
    # Add other known models here

    try:
        # Prefer env var if set and valid
        env_dim_str = os.getenv("MODEL_EMBEDDING_DIM")
        if env_dim_str:
            print(f"DEBUG [insight_engine_core/config.py - get_model_embedding_dim()]: MODEL_EMBEDDING_DIM from env: {env_dim_str}")
            return int(env_dim_str)
        # Else, use default for the determined model name
        print(f"DEBUG [insight_engine_core/config.py - get_model_embedding_dim()]: MODEL_EMBEDDING_DIM not in env, using default for model '{model_name}': {default_dim_for_model}")
        return default_dim_for_model
    except ValueError:
        logger.error(f"MODEL_EMBEDDING_DIM from env ('{os.getenv('MODEL_EMBEDDING_DIM')}') is not an int. Using default {default_dim_for_model}.")
        return default_dim_for_model

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    print(f"DEBUG [insight_engine_core/config.py - get_openai_api_key()]: Called. Key is {'SET' if key else 'NOT SET'}")
    return key

def get_openai_model_name():
    return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

def get_ollama_base_url():
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_ollama_model_name():
    return os.getenv("OLLAMA_MODEL_NAME", "gemma2:9b-instruct-q8_0") # Updated example

def get_hard_split_threshold_chars():
    return int(os.getenv("HARD_SPLIT_THRESHOLD_CHARS", "1000"))

# --- REMOVE ALL MODULE-LEVEL ASSIGNMENTS THAT USE os.getenv() ---
# For example, these should be removed if they were active:
# DATABASE_URL = os.getenv("DATABASE_URL", ...) # REMOVE
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", ...) # REMOVE
# MODEL_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", ...)) # REMOVE
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # REMOVE
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", ...) # REMOVE
# PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY") # REMOVE

# You CAN have true constants here if they don't depend on the environment
# EXAMPLE_CONSTANT = "some_fixed_value"
