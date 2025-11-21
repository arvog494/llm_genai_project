from __future__ import annotations
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Where raw data lives
    data_dir: Path = Path("data/raw")

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3"
    ollama_extraction_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"

    # Vector store
    chroma_db_dir: Path = Path("chroma_db")

    # Source selection / use case
    max_sources_per_use_case: int = 12
    min_relevance_score: float = 0.1
    source_preview_chars: int = 1500
    usecase_graph_dir: Path = Path("graph_sessions")
    usecase_vector_prefix: str = "graphrag_usecase_"

    # Graph settings
    max_neighbors_per_entity: int = 10
    graph_store_path: Path = Path('graph_store.json')
    extraction_max_concurrency: int = 1
    extraction_batch_size: int = 1
    extraction_debug_samples: int = 1
    max_graph_paths_per_answer: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
