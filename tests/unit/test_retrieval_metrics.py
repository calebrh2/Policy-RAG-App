"""Retrieval metric formulas on hand-ranked lists."""

from __future__ import annotations

import math

from rag.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

_K = 5


def test_perfect_top_hit_scores_one_recall_and_ndcg() -> None:
    """A gold chunk at rank 1 of a length-5 list is a perfect ranking."""
    retrieved = ["gold", "b", "c", "d", "e"]
    relevant = {"gold"}

    assert recall_at_k(retrieved, relevant, _K) == 1.0
    assert precision_at_k(retrieved, relevant, _K) == 0.2
    assert reciprocal_rank(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, _K) == 1.0


def test_gold_at_rank_two_discounts_reciprocal_rank_and_ndcg() -> None:
    """Rank 2 yields reciprocal rank 1/2 and NDCG 1/log2(3)."""
    retrieved = ["a", "gold", "c", "d", "e"]
    relevant = {"gold"}

    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, _K) == 1.0 / math.log2(3)


def test_missing_gold_scores_zero() -> None:
    """A ranking with no relevant id scores zero on every metric."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"gold"}

    assert recall_at_k(retrieved, relevant, _K) == 0.0
    assert precision_at_k(retrieved, relevant, _K) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert ndcg_at_k(retrieved, relevant, _K) == 0.0


def test_one_of_two_relevant_ids_in_the_top_k() -> None:
    """One of two relevant ids in the top k is recall 0.5 and precision 1/k."""
    retrieved = ["a", "rel-1", "b", "c", "d"]
    relevant = {"rel-1", "rel-2"}

    assert recall_at_k(retrieved, relevant, _K) == 0.5
    assert precision_at_k(retrieved, relevant, _K) == 1 / _K
    ideal = 1.0 + (1.0 / math.log2(3))
    assert ndcg_at_k(retrieved, relevant, _K) == (1.0 / math.log2(3)) / ideal


def test_score_retrieval_averages_each_metric() -> None:
    """The set score is the mean of the per-query scores."""
    perfect = (["gold", "b", "c", "d", "e"], {"gold"})
    missing = (["a", "b", "c", "d", "e"], {"gold"})

    scores = score_retrieval([perfect, missing], _K)

    assert scores.k == _K
    assert scores.recall_at_k == 0.5
    assert scores.precision_at_k == 0.1
    assert scores.mrr == 0.5
    assert scores.ndcg_at_k == 0.5
