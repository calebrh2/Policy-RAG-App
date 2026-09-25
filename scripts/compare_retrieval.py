"""Compare dense, BM25, hybrid RRF, and reranked retrieval on the real corpus."""

from __future__ import annotations

import argparse

from rag.adapters.base import SearchResult
from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.reranker import CrossEncoderRerankerAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.config import get_settings
from rag.indexing import load_searchable_records
from rag.retrieval import RetrievalService

DEFAULT_QUESTION = "What must new or renewed vendor contracts include by December 2026?"
DEFAULT_EXPECTED = "single-use-plastic-free-policy:2026-03-10:targets:01:01a70e5c"


def rank_of(results: list[SearchResult], expected_id: str) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, 1) if result.record.id == expected_id),
        None,
    )


def print_ranking(name: str, results: list[SearchResult], expected_id: str) -> None:
    print(f"\n{name}")
    for rank, result in enumerate(results[:5], 1):
        metadata = result.record.metadata
        marker = " ← expected" if result.record.id == expected_id else ""
        print(
            f"{rank}. {metadata['document_title']} | {metadata['section_path']} "
            f"| score={result.score:.4f}{marker}"
        )
    expected_rank = rank_of(results, expected_id)
    print(f"Expected chunk rank: {expected_rank if expected_rank is not None else 'not retrieved'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--expected-chunk", default=DEFAULT_EXPECTED)
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    records = load_searchable_records(settings.chunks_path)
    keyword = BM25KeywordSearchAdapter()
    keyword.index(records)
    service = RetrievalService(
        BgeEmbeddingAdapter(
            model_name=settings.embedding_model,
            application_token_limit=settings.embedding_token_limit,
        ),
        ChromaVectorStoreAdapter(
            settings.chroma_path,
            collection_name=settings.chroma_collection,
        ),
        keyword,
        CrossEncoderRerankerAdapter(model_name=settings.reranker_model),
    )

    dense = service.dense(args.question, limit=args.candidates)
    sparse = service.sparse(args.question, limit=args.candidates)
    hybrid = service.hybrid(
        args.question,
        candidate_limit=args.candidates,
        limit=args.candidates * 2,
    )
    reranked = service.reranker.rerank(args.question, hybrid, limit=5)  # type: ignore[union-attr]

    print("RETRIEVAL STRATEGY COMPARISON")
    print(f"Question: {args.question}")
    print(f"Expected chunk: {args.expected_chunk}")
    print_ranking("DENSE ONLY (BGE)", dense, args.expected_chunk)
    print_ranking("BM25 ONLY", sparse, args.expected_chunk)
    print_ranking("HYBRID (RRF)", hybrid, args.expected_chunk)
    print_ranking("HYBRID + CROSS-ENCODER", reranked, args.expected_chunk)

    dense_rank = rank_of(dense, args.expected_chunk)
    hybrid_rank = rank_of(hybrid, args.expected_chunk)
    improved = hybrid_rank is not None and (dense_rank is None or hybrid_rank < dense_rank)
    print(
        "\nHybrid result: "
        + (
            f"PASS — expected chunk improved from {dense_rank or 'not retrieved'} to {hybrid_rank}."
            if improved
            else f"NO IMPROVEMENT — dense rank={dense_rank}, hybrid rank={hybrid_rank}."
        )
    )
    if not improved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
