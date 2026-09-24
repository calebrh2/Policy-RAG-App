"""ChromaDB implementation of the vector-store adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rag.adapters.base import MetadataValue, SearchResult, VectorRecord


class ChromaVectorStoreAdapter:
    """Persistent cosine-similarity storage with caller-supplied embeddings."""

    def __init__(
        self,
        persist_path: Path | str = "data/chromadb",
        *,
        collection_name: str = "policy_chunks",
        client: Any | None = None,
    ) -> None:
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _where(filters: Mapping[str, MetadataValue] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = [{key: {"$eq": value}} for key, value in filters.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def upsert(
        self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings must have the same length")
        if not records:
            return
        self.collection.upsert(
            ids=[record.id for record in records],
            documents=[record.text for record in records],
            metadatas=[dict(record.metadata) for record in records],
            embeddings=cast(Any, [list(vector) for vector in embeddings]),
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        result = self.collection.query(
            query_embeddings=cast(Any, [list(query_embedding)]),
            n_results=limit,
            where=self._where(filters),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            SearchResult(
                VectorRecord(id=record_id, text=document or "", metadata=metadata or {}),
                score=1.0 - float(distance),
            )
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def count(self) -> int:
        return int(self.collection.count())
