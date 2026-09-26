"""Sentence Transformers embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.adapters.base import EmbeddingAdapter


class InputTooLongError(ValueError):
    """Raised before a model can silently truncate an embedding input."""


class BgeEmbeddingAdapter:
    """CPU-friendly BGE adapter with provider details kept out of pipelines."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    EXPECTED_DIMENSIONS = 384
    MODEL_MAX_TOKENS = 512
    DEFAULT_APPLICATION_LIMIT = 480

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        application_token_limit: int = DEFAULT_APPLICATION_LIMIT,
        model: Any | None = None,
    ) -> None:
        if not 0 < application_token_limit <= self.MODEL_MAX_TOKENS:
            raise ValueError("application_token_limit must be between 1 and 512")
        self.model_name = model_name
        self.dimensions = self.EXPECTED_DIMENSIONS
        self.max_input_tokens = self.MODEL_MAX_TOKENS
        self.application_token_limit = application_token_limit
        self._model = model

    @property
    def model(self) -> Any:
        """Load model weights only when the adapter is first used."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension != self.dimensions:
                raise ValueError(
                    f"{self.model_name} returned {dimension} dimensions; expected {self.dimensions}"
                )
        return self._model

    def _input_text(self, text: str, *, is_query: bool) -> str:
        """Prepend the BGE query instruction only when embedding a question."""
        return f"{self.QUERY_INSTRUCTION}{text}" if is_query else text

    def count_tokens(self, text: str, *, is_query: bool = False) -> int:
        """Count tokens the real BGE tokenizer would use for this input."""
        prepared = self._input_text(text, is_query=is_query)
        return len(
            self.model.tokenizer.encode(
                prepared,
                add_special_tokens=True,
                truncation=False,
            )
        )

    def _validate(self, text: str, *, is_query: bool) -> str:
        """Return embeddable text, or raise if it would be truncated."""
        prepared = self._input_text(text, is_query=is_query)
        count = self.count_tokens(text, is_query=is_query)
        limit = self.max_input_tokens if is_query else self.application_token_limit
        if count > limit:
            kind = "query" if is_query else "document"
            raise InputTooLongError(
                f"BGE {kind} input has {count} tokens; limit is {limit}. "
                "Split the input before embedding."
            )
        return prepared

    @staticmethod
    def _as_lists(vectors: Any) -> list[list[float]]:
        """Turn model output into plain lists of floats."""
        raw = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        return [[float(value) for value in vector] for vector in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk text. Queries use embed_query so the instruction is added."""
        if not texts:
            return []
        prepared = [self._validate(text, is_query=False) for text in texts]
        vectors = self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_lists(vectors)

    def embed_query(self, query: str) -> list[float]:
        """Embed a question with the BGE search instruction prepended."""
        prepared = self._validate(query, is_query=True)
        vector = self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw = vector.tolist() if hasattr(vector, "tolist") else vector
        return [float(value) for value in raw]


def build_embedding_adapter(name: str, **kwargs: Any) -> EmbeddingAdapter:
    """Composition-root factory; pipelines depend on the protocol, not this choice."""
    if name.casefold() in {"bge", "bge-small-en-v1.5"}:
        return BgeEmbeddingAdapter(**kwargs)
    raise ValueError(f"Unsupported embedding adapter: {name}")
