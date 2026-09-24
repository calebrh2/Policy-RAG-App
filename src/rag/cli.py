"""Command-line interface for policy RAG ingestion, retrieval, and answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.ollama import OllamaAdapter
from rag.adapters.reranker import CrossEncoderRerankerAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.chunk_validation import validate_chunks
from rag.generation import GroundedAnswerGenerator
from rag.indexing import index_records, load_searchable_records
from rag.ingestion import Chunk, create_chunks
from rag.pipeline import RAGPipeline
from rag.retrieval import RetrievalService
from rag.router import QueryRoute, QueryRouter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN = ROOT / "data/extracted/RAG-documents"
DEFAULT_CHUNKS = ROOT / "data/chunks/chunks.jsonl"
DEFAULT_CHROMA = ROOT / "data/chromadb"
DEFAULT_COLLECTION = "policy_chunks"


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_file(path: Path, embedder: BgeEmbeddingAdapter) -> list[dict[str, Any]]:
    validated = validate_chunks(_load_jsonl(path), embedder)
    path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in validated),
        encoding="utf-8",
    )
    return validated


def _build_retrieval(args: argparse.Namespace) -> RetrievalService:
    records = load_searchable_records(args.chunks)
    bm25 = BM25KeywordSearchAdapter()
    bm25.index(records)
    return RetrievalService(
        BgeEmbeddingAdapter(),
        ChromaVectorStoreAdapter(args.chroma, collection_name=args.collection),
        bm25,
        CrossEncoderRerankerAdapter(),
    )


def command_chunk(args: argparse.Namespace) -> None:
    paths = sorted(args.input_dir.glob("*.md"))
    if not paths:
        raise SystemExit(f"No Markdown files found in {args.input_dir}")
    chunks = [chunk for path in paths for chunk in create_chunks(path)]
    _write_chunks(chunks, args.chunks)
    print(f"Parsed {len(paths)} documents into {len(chunks)} chunks.")
    print(f"Searchable chunks: {sum(chunk.searchable for chunk in chunks)}")
    print(f"Output: {args.chunks}")


def command_validate(args: argparse.Namespace) -> None:
    embedder = BgeEmbeddingAdapter()
    chunks = _validate_file(args.chunks, embedder)
    searchable = [chunk for chunk in chunks if chunk["searchable"]]
    largest = max(searchable, key=lambda chunk: int(chunk["embedding_token_count"]))
    print(f"Validated {len(searchable)} searchable chunks with {embedder.model_name}.")
    print(f"Largest: {largest['embedding_token_count']} tokens ({largest['chunk_id']})")
    print(f"Limit: {embedder.application_token_limit}; oversized: 0")


def command_ingest(args: argparse.Namespace) -> None:
    paths = sorted(args.input_dir.glob("*.md"))
    if paths:
        chunks = [chunk for path in paths for chunk in create_chunks(path)]
        _write_chunks(chunks, args.chunks)
        print(f"Chunked {len(paths)} Markdown documents into {len(chunks)} chunks.")
    elif not args.chunks.exists():
        raise SystemExit(
            f"No Markdown files in {args.input_dir} and no existing chunks at {args.chunks}"
        )
    else:
        print(f"No Markdown inputs found; using existing chunks at {args.chunks}.")

    embedder = BgeEmbeddingAdapter()
    validated = _validate_file(args.chunks, embedder)
    searchable = [chunk for chunk in validated if chunk["searchable"]]
    store = ChromaVectorStoreAdapter(args.chroma, collection_name=args.collection)
    indexed = index_records(
        load_searchable_records(args.chunks),
        embedder,
        store,
        batch_size=args.batch_size,
    )
    print(f"Validated {len(searchable)} searchable chunks.")
    print(f"Indexed {indexed} chunks; collection now contains {store.count()} records.")
    print(f"ChromaDB: {args.chroma} / {args.collection}")


def command_route(args: argparse.Namespace) -> None:
    decision = QueryRouter().route(args.query)
    print(f"route={decision.route.value}")
    print(f"statuses={','.join(decision.statuses)}")
    print(f"reason={decision.reason}")


def _selected_route(value: str) -> QueryRoute | None:
    return None if value == "auto" else QueryRoute(value)


def command_retrieve(args: argparse.Namespace) -> None:
    service = _build_retrieval(args)
    decision = QueryRouter().route(args.query)
    statuses: tuple[str, ...]
    if args.route != "auto":
        decision_route = QueryRoute(args.route)
        statuses = (
            ("current", "superseded")
            if decision_route is QueryRoute.COMPARISON
            else (("superseded",) if decision_route is QueryRoute.HISTORICAL else ("current",))
        )
    else:
        statuses = decision.statuses
    results = service.retrieve_statuses(
        args.query,
        statuses=statuses,
        candidate_limit=args.candidates,
        final_limit=args.top_k,
    )
    print(f"Route: {args.route if args.route != 'auto' else decision.route.value}")
    for rank, result in enumerate(results, 1):
        metadata = result.record.metadata
        print(
            f"{rank}. score={result.score:.4f} {metadata['document_title']} "
            f"version={metadata['version']} section={metadata['section_path']} "
            f"pages={metadata['page_start']}-{metadata['page_end']}"
        )
        print(f"   {result.record.id}")


def command_ask(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        _build_retrieval(args),
        GroundedAnswerGenerator(
            OllamaAdapter(
                model_name=args.model,
                base_url=args.ollama_url,
                context_window=args.context_window,
            )
        ),
    )
    answer, decision = pipeline.answer(
        args.question,
        candidate_limit=args.candidates,
        context_limit=args.top_k,
        route=_selected_route(args.route),
    )
    print(f"Route: {decision.route.value} ({decision.reason})\n")
    print(answer.answer)
    print("\nSources:")
    for citation in answer.citations:
        print(
            f"- {citation.document_title}, version {citation.version}, "
            f"{citation.section_path}, pages {citation.page_start}-{citation.page_end} "
            f"[{citation.chunk_id}]"
        )


def _common_storage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)


def _common_query(parser: argparse.ArgumentParser) -> None:
    _common_storage(parser)
    parser.add_argument(
        "--route",
        choices=("auto", "current", "historical", "comparison"),
        default="auto",
    )
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.cli", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    chunk = commands.add_parser("chunk", help="Create structural chunks from Markdown")
    chunk.add_argument("--input-dir", type=Path, default=DEFAULT_MARKDOWN)
    chunk.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    chunk.set_defaults(handler=command_chunk)

    validate = commands.add_parser("validate", help="Validate chunks with BGE tokenizer")
    validate.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    validate.set_defaults(handler=command_validate)

    ingest = commands.add_parser("ingest", help="Chunk, validate, embed, and index")
    ingest.add_argument("--input-dir", type=Path, default=DEFAULT_MARKDOWN)
    _common_storage(ingest)
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.set_defaults(handler=command_ingest)

    route = commands.add_parser("route", help="Show automatic version routing")
    route.add_argument("query")
    route.set_defaults(handler=command_route)

    retrieve = commands.add_parser("retrieve", help="Retrieve and rerank chunks")
    retrieve.add_argument("query")
    _common_query(retrieve)
    retrieve.set_defaults(handler=command_retrieve)

    ask = commands.add_parser("ask", help="Generate a grounded answer with citations")
    ask.add_argument("question")
    _common_query(ask)
    ask.add_argument("--model", default="mistral:7b")
    ask.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )
    ask.add_argument("--context-window", type=int, default=8192)
    ask.set_defaults(handler=command_ask)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
