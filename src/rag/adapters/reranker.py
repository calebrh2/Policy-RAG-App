"""Cross-encoder adapter for reranking retrieved policy chunks.

Purpose
-------
Defines the interface retrieval uses to score a query against passages, plus
one SentenceTransformer cross-encoder implementation. The model name is chosen
by the caller, so this module does not pick a default model.

Contents
--------
- ``Reranker``: score one query against many passages.
- ``CrossEncoderReranker``: any ``sentence_transformers.CrossEncoder`` model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """A cross-encoder. Higher scores are better matches for the query."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against one query.

        Args:
            query: User text.
            passages: Candidate chunk texts, in candidate order.

        Returns:
            One score per passage. The score is the model's raw logit.
        """
        ...


class CrossEncoderReranker:
    """Passage scores from a SentenceTransformer cross-encoder.

    The model name is chosen by the caller. Scores are raw logits. Ranking
    uses their order, so they are not passed through a sigmoid.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load a cross-encoder.

        Args:
            model_name: Hugging Face or local cross-encoder id.
            device: Torch device. Defaults to CPU.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ValueError(
                "sentence-transformers is not installed. Install the rag group with "
                "`uv sync --group dev --group rag`."
            ) from exc

        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against ``query``.

        Args:
            query: User text.
            passages: Candidate chunk texts.

        Returns:
            One raw logit per passage, in the same order. An empty passage
            list returns an empty score list.
        """
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        raw = self._model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        return [float(value) for value in raw]
