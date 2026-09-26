"""Cross-encoder adapter for reranking retrieved passages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.adapters.base import SearchResult


class CrossEncoderRerankerAdapter:
    """CPU-friendly MS MARCO passage reranker loaded on first use."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_length: int = 512,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        """Load the cross-encoder the first time a candidate list is rescored."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        limit: int,
    ) -> list[SearchResult]:
        """Score each candidate against the question and keep the best ones."""
        if limit <= 0 or not candidates:
            return []
        scores = self.model.predict(
            [(query, candidate.record.text) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [
            SearchResult(candidate.record, float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:limit]
