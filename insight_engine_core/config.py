import os
from dotenv import load_dotenv

# Load environment variables from .env file that should be in the root
# of the application using this library (e.g., niche_hunter_app/.env)
# The library itself shouldn't bundle a .env file.
# Applications using this library are responsible for providing the .env
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Go up 2 levels to trend_analysis_platform
dotenv_path = os.path.join(project_root_path, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback if .env is not in the overarching project root,
    # try loading from the application's root if this library is installed.
    # This part is tricky for a library; usually, the app loads its own .env.
    # For now, we'll assume the app (like niche_hunter_app) will call load_dotenv()
    # or the .env is at the very top level as structured.
    # A better approach for libraries is to expect config to be passed in or env vars to be pre-set.
    print(f"Warning: .env file not found at {dotenv_path}. Relying on pre-set environment variables.")
    load_dotenv() # Tries to load .env from current dir or parent dirs if not found at root


# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")
# Example for local PostgreSQL: "postgresql://youruser:yourpass@localhost:5432/insight_db"

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# For local models, this could also be a path

# --- LLM Configuration (examples, will be expanded) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Add other LLM API keys or local model paths here

# --- Other API Keys ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "InsightEngine/0.1 by YourUsername")


# --- Basic Logging Configuration (can be expanded in utils/logging_setup.py) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


if __name__ == '__main__':
    # For testing the config loading
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not Set'}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
