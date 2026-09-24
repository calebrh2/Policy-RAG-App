import json
from pathlib import Path

import chromadb

from rag.adapters.base import ChunkRecord
from rag.adapters.vector_store import ChromaVectorStore


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return _vector(query)


def test_upsert_and_query_returns_the_nearest_current_chunk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [
            _chunk("current-emissions", "current", "India Scope 2 in 2023 was 6,746.70"),
            _chunk("outdated-target", "outdated", "The target is 15% by 2027"),
            _chunk("plastic", "current", "Eliminate single-use plastic by Dec 2026"),
        ]
    )

    hits = store.query("India Scope 2 6,746.70", version="current")

    assert hits[0].chunk_id == "current-emissions"
    assert hits[0].version == "current"
    assert "6,746.70" in hits[0].text
    assert all(hit.version == "current" for hit in hits)
    assert all(hit.chunk_id != "outdated-target" for hit in hits)


def test_upsert_jsonl_replaces_a_chunk_with_the_same_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "policy::purpose::00",
                "document_name": "Water-Management-Policy",
                "version": "current",
                "section": "Purpose",
                "source_pages": "3-3",
                "text": "Water is a shared resource.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert store.upsert_jsonl(path) == 1
    hits = store.query("shared resource", version="current")

    assert hits[0].section == "Purpose"
    assert hits[0].source_pages == "3-3"
    assert hits[0].document_name == "Water-Management-Policy"


def test_empty_upsert_does_not_call_the_embedder(tmp_path: Path) -> None:
    class _ExplodingEmbedder(_FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError(texts)

    store = ChromaVectorStore(_ExplodingEmbedder(), persist_directory=tmp_path / "chroma")

    store.upsert([])


def _store(tmp_path: Path) -> ChromaVectorStore:
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="policy_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    return ChromaVectorStore(_FakeEmbedder(), collection=collection)


def _chunk(chunk_id: str, version: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version=version,
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text=text,
    )


def _vector(text: str) -> list[float]:
    lowered = text.casefold()
    if "6,746.70" in lowered or "scope 2" in lowered:
        return [1.0, 0.0, 0.0]
    if "plastic" in lowered or "dec 2026" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]
