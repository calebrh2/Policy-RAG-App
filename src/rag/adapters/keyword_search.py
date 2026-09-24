"""Keyword adapter. The pipeline calls this, not Chroma's BM25 function directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

from rag.adapters.base import ChunkRecord, KeywordHit


class _SparseVector(Protocol):
    indices: list[int]
    values: list[float]


class _Bm25Embedder(Protocol):
    def __call__(self, input: list[str]) -> list[_SparseVector]:
        """Embed stored chunk text."""

    def embed_query(self, input: list[str]) -> list[_SparseVector]:
        """Embed one question with the same BM25 function."""


class ChromaBm25Index:
    """Rank chunks with Chroma's BM25 sparse embeddings."""

    def __init__(self) -> None:
        self._records: dict[str, ChunkRecord] = {}
        self._ordered: list[ChunkRecord] = []
        self._vectors: list[_SparseVector] = []
        self._embedder: _Bm25Embedder | None = None

    def upsert(self, records: list[ChunkRecord]) -> None:
        for record in records:
            self._records[record.chunk_id] = record
        self._ordered = list(self._records.values())
        if not self._ordered:
            self._vectors = []
            self._embedder = None
            return
        texts = [record.text for record in self._ordered]
        self._embedder = _embedder(texts)
        self._vectors = list(self._embedder(texts))

    def upsert_jsonl(self, path: Path) -> int:
        records = [
            _record(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.upsert(records)
        return len(records)

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[KeywordHit]:
        if limit < 1 or self._embedder is None:
            return []
        query_vector = self._embedder.embed_query([query])[0]
        ranked = sorted(
            (
                (_dot(query_vector, document), record)
                for document, record in zip(self._vectors, self._ordered, strict=True)
                if (version is None or record.version == version)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [_hit(record, score) for score, record in ranked[:limit] if score > 0]


def _embedder(texts: list[str]) -> _Bm25Embedder:
    from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

    probed = ChromaBm25EmbeddingFunction()(texts)
    average = sum(len(vector.indices) for vector in probed) / len(probed)
    embedder = ChromaBm25EmbeddingFunction(avg_doc_length=average or 1.0)
    return cast(_Bm25Embedder, embedder)


def _dot(query: _SparseVector, document: _SparseVector) -> float:
    weights = dict(zip(document.indices, document.values, strict=True))
    return sum(
        value * weights.get(index, 0.0)
        for index, value in zip(query.indices, query.values, strict=True)
    )


def _record(raw: dict[str, object]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(raw["chunk_id"]),
        document_name=str(raw["document_name"]),
        version=str(raw["version"]),
        section=str(raw["section"]),
        source_pages=str(raw["source_pages"]),
        text=str(raw["text"]),
    )


def _hit(record: ChunkRecord, score: float) -> KeywordHit:
    return KeywordHit(
        chunk_id=record.chunk_id,
        document_name=record.document_name,
        version=record.version,
        section=record.section,
        source_pages=record.source_pages,
        text=record.text,
        score=score,
    )
