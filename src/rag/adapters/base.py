"""Interfaces for the outside systems the pipeline calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk ready to embed and store."""

    chunk_id: str
    document_name: str
    version: str
    section: str
    source_pages: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """One stored chunk returned for a question."""

    chunk_id: str
    document_name: str
    version: str
    section: str
    source_pages: str
    text: str
    distance: float


class Embedder(Protocol):
    """Turns policy text and questions into vectors with the same model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored chunks. Documents are not given the query instruction."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one question so it can be compared with document vectors."""


@dataclass(frozen=True)
class KeywordHit:
    """One stored chunk returned by a keyword search."""

    chunk_id: str
    document_name: str
    version: str
    section: str
    source_pages: str
    text: str
    score: float


class VectorStore(Protocol):
    """Stores chunk vectors and returns the nearest chunks for a question."""

    def upsert(self, records: list[ChunkRecord]) -> None:
        """Embed and save chunks. A repeated chunk_id replaces the old row."""

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[SearchHit]:
        """Return the nearest stored chunks, optionally limited to one version."""


class KeywordIndex(Protocol):
    """Stores chunk text and returns chunks that share the question's exact words."""

    def upsert(self, records: list[ChunkRecord]) -> None:
        """Index chunk text. A repeated chunk_id replaces the old row."""

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[KeywordHit]:
        """Return chunks with the highest keyword score, optionally for one version."""
