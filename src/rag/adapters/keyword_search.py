"""In-memory BM25 keyword-search adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rag.adapters.base import MetadataValue, SearchResult, VectorRecord

TOKEN_RE = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)


def bm25_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25KeywordSearchAdapter:
    """Lexical retrieval for exact policy terms, dates, codes, and numbers."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []
        self._index: BM25Okapi | None = None

    def index(self, records: Sequence[VectorRecord]) -> None:
        self._records = list(records)
        corpus = [bm25_tokens(record.text) for record in self._records]
        self._index = BM25Okapi(corpus) if corpus else None

    @staticmethod
    def _matches(
        record: VectorRecord, filters: Mapping[str, MetadataValue] | None
    ) -> bool:
        return not filters or all(record.metadata.get(key) == value for key, value in filters.items())

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0 or self._index is None:
            return []
        scores = self._index.get_scores(bm25_tokens(query))
        ranked = sorted(
            (
                SearchResult(record, float(score))
                for record, score in zip(self._records, scores, strict=True)
                if score > 0 and self._matches(record, filters)
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:limit]
