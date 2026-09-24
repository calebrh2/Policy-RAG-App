"""Validate generated chunks with BGE's real tokenizer and rewrite the JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.adapters.embeddings import BgeEmbeddingAdapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/chunks/chunks.jsonl"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_chunks(
    chunks: list[dict[str, Any]], adapter: BgeEmbeddingAdapter
) -> list[dict[str, Any]]:
    """Attach exact token counts and reject searchable inputs that would be too long."""
    oversized: list[tuple[str, int]] = []
    validated: list[dict[str, Any]] = []

    for chunk in chunks:
        count = adapter.count_tokens(str(chunk["retrieval_text"]))
        item = dict(chunk)
        item["embedding_token_count"] = count
        validated.append(item)
        if bool(chunk["searchable"]) and count > adapter.application_token_limit:
            oversized.append((str(chunk["chunk_id"]), count))

    if oversized:
        details = "\n".join(f"- {chunk_id}: {count}" for chunk_id, count in oversized)
        raise ValueError(
            "Searchable chunks exceed the BGE application limit and must be "
            f"structurally split before indexing:\n{details}"
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    adapter = BgeEmbeddingAdapter()
    chunks = validate_chunks(load_chunks(args.chunks), adapter)
    args.chunks.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    searchable = [chunk for chunk in chunks if chunk["searchable"]]
    largest = max(searchable, key=lambda chunk: int(chunk["embedding_token_count"]))
    print(f"Validated {len(searchable)} searchable chunks with {adapter.model_name}.")
    print(f"Application limit: {adapter.application_token_limit} tokens")
    print(
        "Largest searchable chunk: "
        f"{largest['embedding_token_count']} tokens ({largest['chunk_id']})"
    )
    print("Oversized searchable chunks: 0")
    print(f"Regenerated: {args.chunks}")


if __name__ == "__main__":
    main()
