"""Embedding adapter. The pipeline calls this, not sentence-transformers directly."""

from __future__ import annotations

from typing import Protocol, cast

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class _Encoder(Protocol):
    def encode(self, texts: list[str], *, normalize_embeddings: bool = False) -> object:
        """Return one vector per text."""


class SentenceTransformerEmbedder:
    """Load one local model and reuse it for chunks and questions."""

    def __init__(self, model_name: str = MODEL_NAME, encoder: _Encoder | None = None) -> None:
        self._model_name = model_name
        self._encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        vectors = self._encode([f"{QUERY_INSTRUCTION}{query}"])
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model().encode(texts, normalize_embeddings=True)
        return [_as_floats(row) for row in encoded]  # type: ignore[attr-defined]

    def _model(self) -> _Encoder:
        encoder = self._encoder
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = cast(_Encoder, SentenceTransformer(self._model_name))
            self._encoder = encoder
        return encoder


def _as_floats(row: object) -> list[float]:
    return [float(value) for value in row]  # type: ignore[attr-defined]
