"""Compare vector-only search with hybrid retrieval on the golden questions.

Run from the repository root, after that corpus has been ingested:

    uv run python scripts/compare_retrieval.py
    uv run python scripts/compare_retrieval.py --corpus coforge

Each question is searched two ways. Vector-only is the dense Chroma ranking.
Hybrid is reciprocal-rank fusion of that ranking and BM25. The script prints
the gold chunk's rank in each list. A lower rank is better. The reranker and
the chat model are not called.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_retrieval import gold_chunk_id

from rag.ChromaDB import get_collection
from rag.corpus import add_corpus_argument, resolve_corpus
from rag.keyword_index import KeywordIndex
from rag.metrics import score_retrieval
from rag.retrieve import ChunkSearch, retrieve
from tests.evaluation.catalog import cases_for

TOP_K = 10


def vector_only_ids(query: str, collection: ChunkSearch, limit: int) -> list[str]:
    """Return dense-search chunk ids, best first.

    The metadata filter matches ``retrieve``: searchable chunks from the
    current edition only.

    Args:
        query: User text.
        collection: Dense chunk store.
        limit: Maximum hits.

    Returns:
        Chunk ids. Empty when the collection is empty or ``limit`` is less
        than 1.
    """
    if limit < 1:
        return []
    total = collection.count()
    if total < 1:
        return []
    result = collection.query(
        query_texts=[query],
        n_results=min(limit, total),
        where={"$and": [{"searchable": True}, {"status": "current"}]},
        include=["metadatas"],
    )
    ids = result.get("ids")
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], list):
        return []
    return [chunk_id for chunk_id in ids[0] if isinstance(chunk_id, str)]


def hybrid_ids(
    query: str,
    collection: ChunkSearch,
    keyword_index: KeywordIndex,
    limit: int,
) -> list[str]:
    """Return hybrid-search chunk ids, best first.

    Args:
        query: User text.
        collection: Dense chunk store.
        keyword_index: BM25 index over the same chunk ids.
        limit: Maximum fused hits, and the size of each list before fusion.

    Returns:
        Chunk ids from reciprocal-rank fusion.
    """
    chunks = retrieve(query, collection, keyword_index, limit=limit, per_list=limit)
    return [chunk.chunk_id for chunk in chunks]


def chunk_rank(chunk_id: str, ids: Sequence[str]) -> int | None:
    """Return the one-based position of a chunk id.

    Args:
        chunk_id: Id to find.
        ids: Ranked ids, best first.

    Returns:
        The position, starting at 1. None when the id is absent.
    """
    try:
        return ids.index(chunk_id) + 1
    except ValueError:
        return None


def format_rank(rank: int | None, limit: int) -> str:
    """Format a one-based rank for the terminal.

    Args:
        rank: One-based position, or None when the id is absent.
        limit: Length of the ranked list that was searched.

    Returns:
        The rank as text, or ``not in top {limit}``.
    """
    if rank is None:
        return f"not in top {limit}"
    return str(rank)


def comparison_verdict(vector_rank: int | None, hybrid_rank: int | None) -> str:
    """Say which ranking placed the gold chunk higher.

    Args:
        vector_rank: One-based dense-only rank, or None when the gold chunk
            is outside the list.
        hybrid_rank: One-based hybrid rank, or None when the gold chunk is
            outside the list.

    Returns:
        ``hybrid wins``, ``vector-only wins``, or ``tie``.
    """
    if hybrid_rank is not None and (vector_rank is None or hybrid_rank < vector_rank):
        return "hybrid wins"
    if vector_rank is not None and (hybrid_rank is None or vector_rank < hybrid_rank):
        return "vector-only wins"
    return "tie"


def main() -> None:
    """Rank each golden question with vector-only search and hybrid retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_corpus_argument(parser)
    corpus = resolve_corpus(parser.parse_args().corpus)
    cases = cases_for(corpus.name).cases
    total = len(cases)
    print(
        f"Comparing {corpus.name}: {total} golden questions at k={TOP_K}",
        flush=True,
    )
    print("Loading dense collection", flush=True)
    collection = get_collection(persist_directory=str(corpus.chroma))
    vector_rankings: list[tuple[list[str], set[str]]] = []
    hybrid_rankings: list[tuple[list[str], set[str]]] = []
    hybrid_wins = 0
    vector_wins = 0
    ties = 0
    with KeywordIndex(str(corpus.keyword)) as index:
        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{total}] {case.case_id}", flush=True)
            print(f"  {case.question}", flush=True)
            print(f"  gold: {case.section_path}", flush=True)
            gold_id = gold_chunk_id(case, collection)
            vector_ids = vector_only_ids(case.question, collection, TOP_K)
            fused_ids = hybrid_ids(case.question, collection, index, TOP_K)
            relevant = {gold_id}
            vector_rankings.append((vector_ids, relevant))
            hybrid_rankings.append((fused_ids, relevant))
            vector_rank = chunk_rank(gold_id, vector_ids)
            hybrid_rank = chunk_rank(gold_id, fused_ids)
            verdict = comparison_verdict(vector_rank, hybrid_rank)
            if verdict == "hybrid wins":
                hybrid_wins += 1
            elif verdict == "vector-only wins":
                vector_wins += 1
            else:
                ties += 1
            print(f"  vector-only rank: {format_rank(vector_rank, TOP_K)}", flush=True)
            print(f"  hybrid rank:      {format_rank(hybrid_rank, TOP_K)}", flush=True)
            print(f"  {verdict}", flush=True)
    vector_scores = score_retrieval(vector_rankings, TOP_K)
    hybrid_scores = score_retrieval(hybrid_rankings, TOP_K)
    print(
        f"Means  vector-only mrr={vector_scores.mrr:.3f} "
        f"recall@{TOP_K}={vector_scores.recall_at_k:.3f}",
        flush=True,
    )
    print(
        f"       hybrid mrr={hybrid_scores.mrr:.3f} recall@{TOP_K}={hybrid_scores.recall_at_k:.3f}",
        flush=True,
    )
    print(
        f"hybrid better: {hybrid_wins}  vector-only better: {vector_wins}  tie: {ties}",
        flush=True,
    )


if __name__ == "__main__":
    main()
