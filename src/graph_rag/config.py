from __future__ import annotations
from pydantic_settings import BaseSettings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Where raw data lives
    data_dir: Path = PROJECT_ROOT / "data" / "raw"

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3"
    ollama_extraction_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"

    # Optional Ollama runtime controls (helpful to avoid VRAM staying "stuck" during evals)
    # If unset, defaults are decided by Ollama / langchain_ollama.
    ollama_keep_alive: str | None = None
    ollama_num_ctx: int | None = None
    ollama_num_predict: int | None = None

    # Vector store
    chroma_db_dir: Path = PROJECT_ROOT / "chroma_db"

    # Source selection / use case
    max_sources_per_use_case: int = 12
    min_relevance_score: float = 0.1
    source_preview_chars: int = 1500
    usecase_graph_dir: Path = PROJECT_ROOT / "graph_sessions"
    usecase_vector_prefix: str = "graphrag_usecase_"

    # Graph settings
    max_neighbors_per_entity: int = 10
    graph_store_path: Path = PROJECT_ROOT / "graph_store.json"
    # Increase concurrency and batch size to speed up extraction/use-case builds.
    # Tune these based on your machine and Ollama server capacity.
    extraction_max_concurrency: int = 8
    extraction_batch_size: int = 8
    # Disable extraction debug samples by default to avoid I/O overhead
    extraction_debug_samples: int = 0
    max_graph_paths_per_answer: int = 5

    class Config:
        env_file = ".venv"


settings = Settings()
