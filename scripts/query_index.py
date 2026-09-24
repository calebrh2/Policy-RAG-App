"""Run hybrid retrieval and cross-encoder reranking over the policy corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.reranker import CrossEncoderRerankerAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.indexing import load_searchable_records
from rag.retrieval import RetrievalService

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks/chunks.jsonl")
    parser.add_argument("--persist-path", type=Path, default=ROOT / "data/chromadb")
    parser.add_argument("--collection", default="policy_chunks")
    parser.add_argument("--status", choices=("current", "superseded"), default="current")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    records = load_searchable_records(args.chunks)
    keyword_search = BM25KeywordSearchAdapter()
    keyword_search.index(records)
    service = RetrievalService(
        BgeEmbeddingAdapter(),
        ChromaVectorStoreAdapter(args.persist_path, collection_name=args.collection),
        keyword_search,
        CrossEncoderRerankerAdapter(),
    )
    results = service.retrieve(
        args.query,
        candidate_limit=args.candidates,
        final_limit=args.top_k,
        filters={"status": args.status},
    )

    for rank, result in enumerate(results, 1):
        metadata = result.record.metadata
        print(
            f"{rank}. score={result.score:.4f} "
            f"document={metadata['document_title']} "
            f"version={metadata['version']} "
            f"section={metadata['section_path']} "
            f"pages={metadata['page_start']}-{metadata['page_end']}"
        )
        print(f"   chunk_id={result.record.id}")
        print(f"   {metadata['original_text'][:500].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
