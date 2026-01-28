"""
Environment variables and model configuration.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root (parent of backend directory)
# Project root is the standard location for monorepos
backend_dir = Path(__file__).parent.parent
project_root = backend_dir.parent
env_path = project_root / ".env"

# Load from project root/.env only
# Use override=False to not override existing environment variables
try:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Try default behavior (current directory) as last resort
        load_dotenv(override=False)
except Exception as e:
    # If loading fails, continue without .env file
    # Environment variables can still be set manually
    print(f"WARNING: Failed to load .env file from {env_path}: {e}")
    print("Continuing without .env file. Environment variables can be set manually.")


# Database configuration
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/coursepilot"
)

# OpenAI configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Embedding configuration
EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small dimension
