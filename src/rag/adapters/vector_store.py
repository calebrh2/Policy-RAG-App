"""ChromaDB adapter. The pipeline calls this, not the Chroma client directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

from rag.adapters.base import ChunkRecord, Embedder, SearchHit

COLLECTION_NAME = "policy_chunks"


class _Collection(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> object:
        """Insert or replace rows."""

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Return the nearest rows."""


class ChromaVectorStore:
    """Embed chunks with the embedding adapter and keep them in ChromaDB."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        persist_directory: Path | None = None,
        collection_name: str = COLLECTION_NAME,
        collection: _Collection | None = None,
    ) -> None:
        self._embedder = embedder
        self._collection = collection
        self._persist_directory = persist_directory
        self._collection_name = collection_name

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        vectors = self._embedder.embed_documents([record.text for record in records])
        self._collection_or_create().upsert(
            ids=[record.chunk_id for record in records],
            embeddings=vectors,
            documents=[record.text for record in records],
            metadatas=[_metadata(record) for record in records],
        )

    def upsert_jsonl(self, path: Path) -> int:
        records = [
            _record(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.upsert(records)
        return len(records)

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[SearchHit]:
        if limit < 1:
            return []
        collection = self._collection_or_create()
        vector = [self._embedder.embed_query(query)]
        if version is None:
            found = collection.query(
                query_embeddings=vector,
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        else:
            found = collection.query(
                query_embeddings=vector,
                n_results=limit,
                include=["documents", "metadatas", "distances"],
                where={"version": version},
            )
        return _hits(found)

    def _collection_or_create(self) -> _Collection:
        collection = self._collection
        if collection is None:
            import chromadb

            if self._persist_directory is None:
                message = "persist_directory is required to open ChromaDB"
                raise ValueError(message)
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self._persist_directory))
            collection = cast(
                _Collection,
                client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                ),
            )
            self._collection = collection
        return collection


def _record(raw: dict[str, object]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(raw["chunk_id"]),
        document_name=str(raw["document_name"]),
        version=str(raw["version"]),
        section=str(raw["section"]),
        source_pages=str(raw["source_pages"]),
        text=str(raw["text"]),
    )


def _metadata(record: ChunkRecord) -> dict[str, str]:
    return {
        "document_name": record.document_name,
        "version": record.version,
        "section": record.section,
        "source_pages": record.source_pages,
    }


def _hits(found: dict[str, object]) -> list[SearchHit]:
    ids = _column(found.get("ids"))
    documents = _column(found.get("documents"))
    metadatas = _column(found.get("metadatas"))
    distances = _column(found.get("distances"))
    hits: list[SearchHit] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        meta = metadata if isinstance(metadata, dict) else {}
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else 0.0
        hits.append(
            SearchHit(
                chunk_id=str(chunk_id),
                document_name=str(meta.get("document_name", "")),
                version=str(meta.get("version", "")),
                section=str(meta.get("section", "")),
                source_pages=str(meta.get("source_pages", "")),
                text=str(document),
                distance=float(distance) if isinstance(distance, int | float) else 0.0,
            )
        )
    return hits


def _column(value: object) -> list[object]:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return []
    return list(value[0])
