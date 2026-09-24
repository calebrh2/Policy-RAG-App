"""Reranking reorders fused candidates by cross-encoder score."""

from __future__ import annotations

from collections.abc import Sequence

from rag.rerank import rerank
from rag.retrieve import RetrievedChunk


class FakeReranker:
    """Scores passages from a map of text to logit."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.queries.append(query)
        return [self.scores[passage] for passage in passages]


def test_rerank_prefers_a_higher_cross_encoder_score() -> None:
    fusion_order = [
        _chunk("fusion-first", "plastic policy boilerplate"),
        _chunk("fusion-second", "plastic cups are banned"),
    ]

    hits = rerank(
        "plastic cups",
        fusion_order,
        FakeReranker(
            {
                "plastic policy boilerplate": 0.1,
                "plastic cups are banned": 4.2,
            }
        ),
    )

    assert [hit.chunk_id for hit in hits] == ["fusion-second", "fusion-first"]
    assert hits[0].score == 4.2
    assert hits[0].text == "plastic cups are banned"
    assert hits[0].metadata == {"status": "current"}


def test_rerank_keeps_only_the_limit() -> None:
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    hits = rerank("query", chunks, FakeReranker({"a": 1.0, "b": 3.0, "c": 2.0}), limit=2)

    assert [hit.chunk_id for hit in hits] == ["b", "c"]


def test_rerank_breaks_equal_scores_by_chunk_id() -> None:
    chunks = [_chunk("b", "same"), _chunk("a", "same")]
    hits = rerank("query", chunks, FakeReranker({"same": 1.0}))

    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert hits[0].score == hits[1].score


def test_rerank_returns_nothing_for_an_empty_candidate_list() -> None:
    assert rerank("query", [], FakeReranker({})) == []


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"status": "current"},
        score=0.0,
    )
