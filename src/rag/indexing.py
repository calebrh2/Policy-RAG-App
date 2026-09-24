"""Load validated chunks, embed them, and index them for retrieval."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag.adapters.base import (
    EmbeddingAdapter,
    MetadataValue,
    VectorRecord,
    VectorStoreAdapter,
)


def load_searchable_records(path: Path) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        chunk: dict[str, Any] = json.loads(line)
        if not chunk["searchable"]:
            continue
        count = chunk.get("embedding_token_count")
        if not isinstance(count, int):
            raise TypeError(
                f"{chunk['chunk_id']} has not been validated with the embedding tokenizer"
            )
        metadata: dict[str, MetadataValue] = {
            "document_id": str(chunk["document_id"]),
            "document_title": str(chunk["document_title"]),
            "source_file": str(chunk["source_file"]),
            "version": str(chunk["version"]),
            "status": str(chunk["status"]),
            "section_path": " > ".join(chunk["section_path"]),
            "page_start": int(chunk["page_start"]),
            "page_end": int(chunk["page_end"]),
            "content_types": ",".join(chunk["content_types"]),
            "embedding_token_count": count,
            "original_text": str(chunk["text"]),
        }
        records.append(
            VectorRecord(
                id=str(chunk["chunk_id"]),
                text=str(chunk["retrieval_text"]),
                metadata=metadata,
            )
        )
    return records


def _batches(records: list[VectorRecord], size: int) -> Iterable[list[VectorRecord]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def index_records(
    records: list[VectorRecord],
    embedder: EmbeddingAdapter,
    vector_store: VectorStoreAdapter,
    *,
    batch_size: int = 16,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indexed = 0
    for batch in _batches(records, batch_size):
        embeddings = embedder.embed_documents([record.text for record in batch])
        vector_store.upsert(batch, embeddings)
        indexed += len(batch)
    return indexed
