"""Interfaces for the outside systems the pipeline calls."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Turns policy text and questions into vectors with the same model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored chunks. Documents are not given the query instruction."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one question so it can be compared with document vectors."""
