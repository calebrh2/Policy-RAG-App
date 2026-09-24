"""Combine Chroma and BM25 ranks into one list."""

from __future__ import annotations

from dataclasses import dataclass

from rag.adapters.base import KeywordHit, KeywordIndex, SearchHit, VectorStore
from rag.adapters.reranker import Reranker

RRF_K = 60


@dataclass(frozen=True)
class RetrievalHit:
    """One chunk after Chroma and BM25 ranks are merged."""

    chunk_id: str
    document_name: str
    version: str
    section: str
    source_pages: str
    text: str
    score: float


class HybridRetriever:
    """Ask both indexes, then merge them by reciprocal rank fusion."""

    def __init__(
        self,
        store: VectorStore,
        keywords: KeywordIndex,
        *,
        candidates: int = 20,
    ) -> None:
        self._store = store
        self._keywords = keywords
        self._candidates = candidates

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        version: str | None = "current",
    ) -> list[RetrievalHit]:
        breadth = max(self._candidates, limit)
        dense = self._store.query(query, limit=breadth, version=version)
        sparse = self._keywords.query(query, limit=breadth, version=version)
        return fuse(dense, sparse, limit=limit)


def rerank(
    query: str,
    hits: list[RetrievalHit],
    reranker: Reranker,
    *,
    limit: int = 5,
) -> list[RetrievalHit]:
    """Order fused chunks by the cross-encoder score and keep the top ones."""
    if limit < 1 or not hits:
        return []
    scores = reranker.score(query, [hit.text for hit in hits])
    ordered = sorted(
        zip(scores, hits, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    return [_with_score(hit, score) for score, hit in ordered[:limit]]


def _with_score(hit: RetrievalHit, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=hit.chunk_id,
        document_name=hit.document_name,
        version=hit.version,
        section=hit.section,
        source_pages=hit.source_pages,
        text=hit.text,
        score=score,
    )


def fuse(
    dense: list[SearchHit],
    sparse: list[KeywordHit],
    *,
    limit: int,
) -> list[RetrievalHit]:
    """Score each chunk by the sum of 1 / (60 + rank) from each list."""
    if limit < 1:
        return []
    scores: dict[str, float] = {}
    chosen: dict[str, RetrievalHit] = {}
    _add_dense(scores, chosen, dense)
    _add_sparse(scores, chosen, sparse)
    ranked = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            document_name=hit.document_name,
            version=hit.version,
            section=hit.section,
            source_pages=hit.source_pages,
            text=hit.text,
            score=scores[chunk_id],
        )
        for chunk_id in ranked[:limit]
        if (hit := chosen[chunk_id])
    ]


def _add_dense(
    scores: dict[str, float],
    chosen: dict[str, RetrievalHit],
    hits: list[SearchHit],
) -> None:
    for rank, hit in enumerate(hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chosen.setdefault(hit.chunk_id, _from_dense(hit))


def _add_sparse(
    scores: dict[str, float],
    chosen: dict[str, RetrievalHit],
    hits: list[KeywordHit],
) -> None:
    for rank, hit in enumerate(hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chosen.setdefault(hit.chunk_id, _from_sparse(hit))


def _from_dense(hit: SearchHit) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=hit.chunk_id,
        document_name=hit.document_name,
        version=hit.version,
        section=hit.section,
        source_pages=hit.source_pages,
        text=hit.text,
        score=0.0,
    )


def _from_sparse(hit: KeywordHit) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=hit.chunk_id,
        document_name=hit.document_name,
        version=hit.version,
        section=hit.section,
        source_pages=hit.source_pages,
        text=hit.text,
        score=0.0,
    )
