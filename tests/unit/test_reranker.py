from __future__ import annotations

from rag.adapters.base import SearchResult, VectorRecord
from rag.adapters.reranker import CrossEncoderRerankerAdapter


class FakeCrossEncoder:
    def predict(self, pairs, *, batch_size, show_progress_bar):  # type: ignore[no-untyped-def]
        assert batch_size == 2
        assert show_progress_bar is False
        return [0.1 if "irrelevant" in passage else 0.9 for _, passage in pairs]


def test_cross_encoder_reranks_and_limits_candidates() -> None:
    candidates = [
        SearchResult(VectorRecord("bad", "irrelevant passage", {}), 10.0),
        SearchResult(VectorRecord("good", "matching passage", {}), 1.0),
    ]
    reranker = CrossEncoderRerankerAdapter(model=FakeCrossEncoder(), batch_size=2)

    results = reranker.rerank("query", candidates, limit=1)

    assert [result.record.id for result in results] == ["good"]
    assert results[0].score == 0.9


def test_cross_encoder_handles_empty_or_zero_limit() -> None:
    reranker = CrossEncoderRerankerAdapter(model=FakeCrossEncoder())
    assert reranker.rerank("query", [], limit=5) == []
    candidate = SearchResult(VectorRecord("id", "text", {}), 1.0)
    assert reranker.rerank("query", [candidate], limit=0) == []
