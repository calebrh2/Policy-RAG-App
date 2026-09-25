"""Screenshot-ready proof of the minimal embed-store-retrieve loop."""

from __future__ import annotations

import chromadb

from rag.adapters.base import VectorRecord
from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.config import get_settings


def main() -> None:
    settings = get_settings()
    embedder = BgeEmbeddingAdapter(
        model_name=settings.embedding_model,
        application_token_limit=settings.embedding_token_limit,
    )
    store = ChromaVectorStoreAdapter(
        client=chromadb.EphemeralClient(),
        collection_name="minimal_retrieval_demo",
    )
    records = [
        VectorRecord(
            id="expense-policy",
            text="Employees must submit expense reports within 30 days of purchase.",
            metadata={"topic": "expenses", "status": "current"},
        ),
        VectorRecord(
            id="remote-work-policy",
            text="Employees may work remotely for up to two days each week.",
            metadata={"topic": "remote work", "status": "current"},
        ),
    ]
    vectors = embedder.embed_documents([record.text for record in records])
    store.upsert(records, vectors)

    query = "How long do employees have to submit an expense report?"
    results = store.search(embedder.embed_query(query), limit=2)

    print("MINIMAL EMBED → STORE → RETRIEVE")
    print(f"Embedding model: {embedder.model_name}")
    print(f"Vector dimensions: {embedder.dimensions}")
    print(f"Stored records: {store.count()}")
    print(f"Query: {query}\n")
    for rank, result in enumerate(results, 1):
        print(f"{rank}. {result.record.id} (cosine similarity={result.score:.4f})")
        print(f"   {result.record.text}")

    passed = bool(results) and results[0].record.id == "expense-policy"
    print(f"\n{'PASS' if passed else 'FAIL'}: expected expense-policy ranked first")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
