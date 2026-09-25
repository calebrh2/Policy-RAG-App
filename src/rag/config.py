"""Central runtime configuration for the policy RAG application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RAGSettings(BaseSettings):
    """Validated defaults that may be overridden through environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="RAG_",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    markdown_dir: Path = PROJECT_ROOT / "data/extracted/RAG-documents"
    chunks_path: Path = PROJECT_ROOT / "data/chunks/chunks.jsonl"
    chroma_path: Path = PROJECT_ROOT / "data/chromadb"
    chroma_collection: str = "policy_chunks"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_token_limit: int = Field(default=480, ge=1, le=512)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    retrieval_candidates: int = Field(default=10, ge=1)
    retrieval_top_k: int = Field(default=3, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)

    ollama_model: str = Field(
        default="qwen3:8b",
        validation_alias=AliasChoices("RAG_OLLAMA_MODEL", "OLLAMA_MODEL", "LLM_MODEL"),
    )
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        validation_alias=AliasChoices("RAG_OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "LLM_BASE_URL"),
    )
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_context_window: int = Field(default=8192, ge=512)
    ollama_think: bool = False
    router_confidence: float = Field(default=0.70, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> RAGSettings:
    """Load and cache the validated application configuration."""

    return RAGSettings()
