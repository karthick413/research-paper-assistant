
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_DIR / "app"
DATA_DIR = PROJECT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
STORAGE_DIR = PROJECT_DIR / "storage"

# Ollama models
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text:latest"

# Text chunking
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180

# Retrieval
TOP_K = 5
SUMMARY_TOP_K = 7

# Embedding batch size
EMBED_BATCH_SIZE = 16

# Ensure required project directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
