"""Embed validated policy chunks and upsert them into persistent ChromaDB."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.indexing import index_records, load_searchable_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks/chunks.jsonl")
    parser.add_argument("--persist-path", type=Path, default=ROOT / "data/chromadb")
    parser.add_argument("--collection", default="policy_chunks")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    records = load_searchable_records(args.chunks)
    embedder = BgeEmbeddingAdapter()
    store = ChromaVectorStoreAdapter(args.persist_path, collection_name=args.collection)
    indexed = index_records(records, embedder, store, batch_size=args.batch_size)
    print(f"Indexed {indexed} searchable chunks into {args.collection}.")
    print(f"Collection records: {store.count()}")
    print(f"Persistent path: {args.persist_path}")


if __name__ == "__main__":
    main()
