from __future__ import annotations

from rag.adapters.base import SearchResult, VectorRecord
from rag.retrieval import reciprocal_rank_fusion


def result(record_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(VectorRecord(record_id, record_id, {}), score)


def test_rrf_rewards_results_present_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [
            [result("dense-only"), result("both")],
            [result("both"), result("sparse-only")],
        ],
        limit=3,
    )

    assert fused[0].record.id == "both"
    assert {item.record.id for item in fused} == {"both", "dense-only", "sparse-only"}


def test_rrf_deduplicates_and_respects_limit() -> None:
    fused = reciprocal_rank_fusion([[result("a"), result("b")], [result("a")]], limit=1)
    assert [item.record.id for item in fused] == ["a"]
