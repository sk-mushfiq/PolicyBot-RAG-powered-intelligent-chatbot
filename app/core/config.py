"""
app/core/config.py
──────────────────
Centralised settings loaded from .env via Pydantic BaseSettings.
All modules import from here — never from os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with type validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "HR PolicyBot"
    app_version: str = "1.0.0"
    debug: bool = False

    # OpenAI
    openai_api_key: str = "your-key-here"

    # Models
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    embedding_provider: str = "openai"          # "openai" | "huggingface"
    huggingface_model: str = "all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 4
    max_tokens: int = 1000

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "hr_policies"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()
