from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import RAGSettings


def test_config_has_project_runtime_defaults() -> None:
    settings = RAGSettings()

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_token_limit == 480
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert settings.retrieval_top_k == 3
    assert settings.ollama_think is False
    assert settings.ollama_model == "qwen3:8b"
    assert settings.chroma_collection == "policy_chunks"


def test_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        RAGSettings(embedding_token_limit=513)

    with pytest.raises(ValidationError):
        RAGSettings(router_confidence=1.1)
