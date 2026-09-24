from __future__ import annotations

import chromadb
import pytest

from rag.adapters.base import VectorRecord
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter


def record(record_id: str, text: str, status: str = "current") -> VectorRecord:
    return VectorRecord(record_id, text, {"status": status, "version": "v1"})


def test_chroma_upsert_search_and_filter() -> None:
    store = ChromaVectorStoreAdapter(
        client=chromadb.EphemeralClient(),
        collection_name="adapter_test",
    )
    records = [
        record("current", "current carbon target"),
        record("old", "old carbon target", "superseded"),
    ]
    store.upsert(records, [[1.0, 0.0], [0.9, 0.1]])

    results = store.search([1.0, 0.0], limit=5, filters={"status": "current"})

    assert store.count() == 2
    assert [result.record.id for result in results] == ["current"]
    assert results[0].score == pytest.approx(1.0)


def test_chroma_upsert_is_idempotent() -> None:
    store = ChromaVectorStoreAdapter(
        client=chromadb.EphemeralClient(),
        collection_name="idempotent_test",
    )
    records = [record("same-id", "first")]
    store.upsert(records, [[1.0, 0.0]])
    store.upsert([record("same-id", "updated")], [[0.0, 1.0]])

    assert store.count() == 1
    result = store.search([0.0, 1.0], limit=1)[0]
    assert result.record.text == "updated"


def test_bm25_finds_exact_terms_and_filters_versions() -> None:
    search = BM25KeywordSearchAdapter()
    search.index(
        [
            record("current", "Electric vehicle integration target is 50 percent by 2030."),
            record("old", "Electric vehicle integration target was 15 percent by 2027.", "superseded"),
            record("water", "Reduce water consumption across offices."),
        ]
    )

    results = search.search(
        "electric vehicle 2030",
        limit=3,
        filters={"status": "current"},
    )

    assert [result.record.id for result in results] == ["current"]


def test_adapters_reject_or_ignore_empty_work() -> None:
    store = ChromaVectorStoreAdapter(
        client=chromadb.EphemeralClient(),
        collection_name="empty_test",
    )
    store.upsert([], [])
    assert store.search([1.0, 0.0], limit=0) == []
    with pytest.raises(ValueError, match="same length"):
        store.upsert([record("one", "text")], [])

    keyword = BM25KeywordSearchAdapter()
    keyword.index([])
    assert keyword.search("anything", limit=5) == []
