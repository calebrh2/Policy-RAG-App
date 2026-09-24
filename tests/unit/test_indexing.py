from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.adapters.base import VectorRecord
from rag.indexing import index_records, load_searchable_records


class FakeEmbedder:
    model_name = "fake"
    dimensions = 2
    max_input_tokens = 512
    application_token_limit = 480

    def count_tokens(self, text: str, *, is_query: bool = False) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query)), 1.0]


class FakeStore:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def upsert(self, records: list[VectorRecord], embeddings: list[list[float]]) -> None:
        assert len(records) == len(embeddings)
        self.ids.extend(record.id for record in records)

    def search(self, query_embedding: list[float], *, limit: int, filters=None):  # type: ignore[no-untyped-def]
        return []


def write_chunks(path: Path, *, include_count: bool = True) -> None:
    searchable = {
        "chunk_id": "searchable",
        "document_id": "doc",
        "document_title": "Document",
        "source_file": "doc.pdf",
        "version": "v1",
        "status": "current",
        "section_path": ["Section"],
        "page_start": 1,
        "page_end": 2,
        "content_types": ["paragraph"],
        "searchable": True,
        "text": "Original",
        "retrieval_text": "Document Section Original",
    }
    if include_count:
        searchable["embedding_token_count"] = 3
    ignored = dict(searchable, chunk_id="ignored", searchable=False)
    path.write_text(
        json.dumps(searchable) + "\n" + json.dumps(ignored) + "\n",
        encoding="utf-8",
    )


def test_load_and_index_only_validated_searchable_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_chunks(path)
    records = load_searchable_records(path)
    store = FakeStore()

    count = index_records(records, FakeEmbedder(), store, batch_size=1)

    assert count == 1
    assert store.ids == ["searchable"]
    assert records[0].metadata["section_path"] == "Section"


def test_loader_requires_embedding_validation(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, include_count=False)

    with pytest.raises(TypeError, match="has not been validated"):
        load_searchable_records(path)
