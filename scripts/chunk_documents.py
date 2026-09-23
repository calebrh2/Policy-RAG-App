"""Write structural chunks for the extracted policy Markdown.

    uv run python scripts/chunk_documents.py

Reads data/extracted/RAG-documents and writes data/chunks/preview.md plus
data/chunks/chunks.jsonl. Running it again overwrites those files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking import chunk_directory


def main() -> None:
    source = ROOT / "data" / "extracted" / "RAG-documents"
    output = ROOT / "data" / "chunks"
    if not source.is_dir():
        raise SystemExit(f"No Markdown directory at {source}")
    chunks = chunk_directory(source)
    if not chunks:
        raise SystemExit(f"No Markdown files found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    preview: list[str] = ["# Chunk preview", ""]
    records: list[str] = []
    current = ""
    for chunk in chunks:
        if chunk.document_name != current:
            current = chunk.document_name
            preview.extend([f"# {current}", ""])
        preview.extend(
            [
                f"## {chunk.chunk_id}",
                "",
                f"- version: {chunk.version}",
                f"- section: {chunk.section}",
                f"- source_pages: {chunk.source_pages}",
                f"- tokens: {chunk.token_count}",
                "",
                chunk.text,
                "",
                "---",
                "",
            ]
        )
        records.append(
            json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "version": chunk.version,
                    "section": chunk.section,
                    "source_pages": chunk.source_pages,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                },
                ensure_ascii=False,
            )
        )
        print(
            f"{chunk.token_count:4d}  {chunk.version:9}  p{chunk.source_pages or '-':5}  {chunk.section}"
        )

    (output / "preview.md").write_text("\n".join(preview), encoding="utf-8")
    (output / "chunks.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    print(f"\n{len(chunks)} chunks")
    print(output / "preview.md")


if __name__ == "__main__":
    main()
