import os

# This library expects environment variables to be set by the consuming application.
# It can define defaults here if environment variables are not found.

# --- Database Configuration ---
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/default_db_name_if_not_set")
# insight_engine_core/insight_engine_core/config.py
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    # Option A: Raise an error if not set by consuming app
    # raise EnvironmentError("DATABASE_URL environment variable not set by the application.")
    # Option B: Log a warning and proceed (might fail later, as it did)
    print("WARNING: insight_engine_core.config - DATABASE_URL not set. Engine creation might fail.")
    DATABASE_URL = "postgresql://invalid_user:invalid_pass@invalid_host:0000/invalid_db" # A clearly invalid default

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Try to get dimension from an env var, or set a default based on default model
# This is still tricky for model definition, as models.py needs this at import time.
# A common pattern is to require the app to configure this or use a fixed known dimension.
DEFAULT_EMBEDDING_DIM = 384 # For all-MiniLM-L6-v2
MODEL_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIM))

# --- LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Add other LLM API keys or local model paths here

# --- Other API Keys ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
# REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
# REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
# REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "InsightEngine/0.1 by YourUsername")


# --- Basic Logging Configuration (can be expanded in utils/logging_setup.py) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


if __name__ == '__main__':
    # For testing the config loading
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not Set'}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
