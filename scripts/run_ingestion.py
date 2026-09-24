"""Generate adaptive chunks for inspection; this does not embed them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.ingestion import Chunk, create_chunks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/extracted/RAG-documents"
DEFAULT_OUTPUT = ROOT / "data/chunks"


def review_markdown(chunks: list[Chunk]) -> str:
    parts = [
        "# Adaptive chunk review",
        "",
        "> Section-first; preferred 100–450; hard maximum 500; zero sliding overlap.",
        "> Counts are model-independent estimates including retrieval context.",
    ]
    for chunk in chunks:
        parts.extend(
            [
                "",
                f"## {chunk.chunk_id}",
                "",
                f"- Document: {chunk.document_title}",
                f"- Version/status: {chunk.version} / {chunk.status}",
                f"- Section: {' > '.join(chunk.section_path) or 'Document metadata'}",
                f"- Pages: {chunk.page_start}-{chunk.page_end}",
                f"- Types: {', '.join(chunk.content_types)}",
                f"- Searchable: {str(chunk.searchable).lower()}",
                f"- Estimated retrieval tokens: {chunk.estimated_tokens}",
                "",
                "### Stored content",
                "",
                chunk.text,
                "",
                "### Retrieval text",
                "",
                chunk.retrieval_text,
            ]
        )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--target-tokens", type=int, default=350)
    parser.add_argument("--soft-max-tokens", type=int, default=450)
    parser.add_argument("--hard-max-tokens", type=int, default=500)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.md"))
    if not paths:
        parser.error(f"No Markdown files found in {args.input_dir}")
    chunks = [
        chunk
        for path in paths
        for chunk in create_chunks(
            path,
            preferred_min_tokens=args.min_tokens,
            target_tokens=args.target_tokens,
            soft_max_tokens=args.soft_max_tokens,
            hard_max_tokens=args.hard_max_tokens,
        )
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "chunks.jsonl"
    review = args.output_dir / "chunks-review.md"
    jsonl.write_text(
        "".join(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    review.write_text(review_markdown(chunks), encoding="utf-8")
    searchable = [chunk for chunk in chunks if chunk.searchable]
    print(f"Parsed {len(paths)} documents into {len(chunks)} chunks.")
    print(f"Searchable chunks: {len(searchable)}")
    print(
        f"Searchable chunks below preferred minimum: "
        f"{sum(c.estimated_tokens < args.min_tokens for c in searchable)}"
    )
    print(
        f"Chunks above hard maximum: "
        f"{sum(c.estimated_tokens > args.hard_max_tokens for c in chunks)}"
    )
    print(f"JSONL: {jsonl}")
    print(f"Review: {review}")


if __name__ == "__main__":
    main()
