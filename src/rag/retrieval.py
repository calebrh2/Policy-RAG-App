"""Dense, sparse, hybrid, and reranked retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.adapters.base import (
    EmbeddingAdapter,
    KeywordSearchAdapter,
    MetadataValue,
    RerankerAdapter,
    SearchResult,
    VectorStoreAdapter,
)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse rankings without comparing incompatible dense and BM25 scores."""
    if limit <= 0 or rank_constant <= 0:
        return []
    scores: dict[str, float] = {}
    records = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, 1):
            record_id = result.record.id
            records[record_id] = result.record
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rank_constant + rank)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [SearchResult(records[record_id], scores[record_id]) for record_id in ordered]


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingAdapter,
        vector_store: VectorStoreAdapter,
        keyword_search: KeywordSearchAdapter,
        reranker: RerankerAdapter | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.reranker = reranker

    @staticmethod
    def current_filters(
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> dict[str, MetadataValue]:
        combined: dict[str, MetadataValue] = {"status": "current"}
        if filters:
            combined.update(filters)
        return combined

    def dense(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(
            query_embedding,
            limit=limit,
            filters=self.current_filters(filters),
        )

    def sparse(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        return self.keyword_search.search(
            query,
            limit=limit,
            filters=self.current_filters(filters),
        )

    def hybrid(
        self,
        query: str,
        *,
        candidate_limit: int = 10,
        limit: int = 10,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        dense = self.dense(query, limit=candidate_limit, filters=filters)
        sparse = self.sparse(query, limit=candidate_limit, filters=filters)
        return reciprocal_rank_fusion([dense, sparse], limit=limit)

    def retrieve(
        self,
        query: str,
        *,
        candidate_limit: int = 10,
        final_limit: int = 5,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        candidates = self.hybrid(
            query,
            candidate_limit=candidate_limit,
            limit=candidate_limit * 2,
            filters=filters,
        )
        if self.reranker is None:
            return candidates[:final_limit]
        return self.reranker.rerank(query, candidates, limit=final_limit)
