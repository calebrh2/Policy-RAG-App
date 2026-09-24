"""Cross-encoder adapter. The pipeline calls this, not the model directly."""

from __future__ import annotations

from typing import Protocol, cast

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Scores question-and-chunk pairs. A higher score is better evidence."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one score per chunk, in the same order as texts."""


class _Scorer(Protocol):
    def predict(self, pairs: list[list[str]]) -> object:
        """Return one score per question-and-chunk pair."""


class CrossEncoderReranker:
    """Read the question beside each chunk and score that pair."""

    def __init__(self, model_name: str = MODEL_NAME, scorer: _Scorer | None = None) -> None:
        self._model_name = model_name
        self._scorer = scorer

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._model().predict([[query, text] for text in texts])
        return [float(value) for value in raw]  # type: ignore[attr-defined]

    def _model(self) -> _Scorer:
        scorer = self._scorer
        if scorer is None:
            from sentence_transformers import CrossEncoder

            scorer = cast(_Scorer, CrossEncoder(self._model_name))
            self._scorer = scorer
        return scorer
