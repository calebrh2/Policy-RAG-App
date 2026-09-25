"""Retrieval metrics for a ranked list of chunk ids.

Purpose
-------
Scores one ranked list against the chunk ids that contain the answer, then
averages those scores across a set of questions. The evaluation script uses
these functions. They do not search an index.

Contents
--------
- ``RetrievalScores``: mean recall, precision, MRR, and NDCG at one cutoff.
- ``recall_at_k``: share of relevant ids present in the top k.
- ``precision_at_k``: share of the top k that are relevant.
- ``reciprocal_rank``: inverse rank of the first relevant id.
- ``ndcg_at_k``: binary-gain NDCG at k.
- ``score_retrieval``: mean of the four metrics over many rankings.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScores:
    """Mean retrieval scores for one cutoff.

    ``mrr`` is the mean of per-query reciprocal rank. The other fields are
    means of the matching per-query metric.
    """

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Return the fraction of relevant ids that appear in the top k.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Chunk ids that contain the answer.
        k: Cutoff. Positions after k are ignored.

    Returns:
        Hits divided by the number of relevant ids. Zero when there are no
        relevant ids or ``k`` is less than 1.
    """
    relevant_ids = set(relevant)
    if not relevant_ids or k < 1:
        return 0.0
    hits = relevant_ids.intersection(retrieved[:k])
    return len(hits) / len(relevant_ids)


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Return the fraction of the top k that are relevant.

    A list shorter than ``k`` still divides by ``k``. Missing positions count
    as not relevant.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Chunk ids that contain the answer.
        k: Cutoff and denominator.

    Returns:
        Unique relevant hits in the top k, divided by ``k``. Zero when ``k``
        is less than 1.
    """
    if k < 1:
        return 0.0
    hits = set(relevant).intersection(retrieved[:k])
    return len(hits) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Return one divided by the rank of the first relevant id.

    Args:
        retrieved: Ranked chunk ids, best first. Rank 1 is the first id.
        relevant: Chunk ids that contain the answer.

    Returns:
        ``1 / rank`` for the first hit. Zero when no relevant id is present.
    """
    relevant_ids = set(relevant)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Return binary-gain NDCG at k.

    A relevant id at rank ``r`` adds ``1 / log2(r + 1)``. Rank 1 adds 1.
    The ideal list places every relevant id at the front of the top k.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Chunk ids that contain the answer.
        k: Cutoff. Positions after k are ignored.

    Returns:
        Discounted cumulative gain divided by the ideal gain. Zero when there
        are no relevant ids or ``k`` is less than 1.
    """
    relevant_ids = set(relevant)
    if not relevant_ids or k < 1:
        return 0.0
    seen: set[str] = set()
    gain = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant_ids and chunk_id not in seen:
            seen.add(chunk_id)
            gain += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return gain / ideal


def score_retrieval(
    rankings: Sequence[tuple[Sequence[str], Collection[str]]],
    k: int,
) -> RetrievalScores:
    """Average recall, precision, reciprocal rank, and NDCG at one cutoff.

    Args:
        rankings: Pairs of ranked chunk ids and the relevant ids for that
            question.
        k: Cutoff passed to the per-query metrics. Reciprocal rank uses the
            full ranked list.

    Returns:
        The mean of each metric. All means are zero when ``rankings`` is empty.
    """
    if not rankings:
        return RetrievalScores(0.0, 0.0, 0.0, 0.0, k)
    count = len(rankings)
    recall = sum(recall_at_k(retrieved, relevant, k) for retrieved, relevant in rankings)
    precision = sum(precision_at_k(retrieved, relevant, k) for retrieved, relevant in rankings)
    mrr = sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in rankings)
    ndcg = sum(ndcg_at_k(retrieved, relevant, k) for retrieved, relevant in rankings)
    return RetrievalScores(
        recall_at_k=recall / count,
        precision_at_k=precision / count,
        mrr=mrr / count,
        ndcg_at_k=ndcg / count,
        k=k,
    )
