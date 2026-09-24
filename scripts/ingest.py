"""Ingest the extracted policy markdown into Chroma and the keyword index.

Run from the repository root, after ``uv sync --group dev --group rag``:

    uv run python scripts/ingest.py

The four files in ``data/extracted/RAG-documents`` are the input. The keyword
index and Chroma collection are created on the first run and updated in place
after that. A new edition is stored beside the older one. The first run
downloads ``BAAI/bge-small-en-v1.5`` and embeds every chunk.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rag.ChromaDB import get_collection
from rag.ingest import ingest
from rag.keyword_index import KeywordIndex

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/extracted/RAG-documents"
KEYWORD_PATH = ROOT / "data/keyword/chunks.sqlite"
CHROMA_PATH = ROOT / "data/chromadb"


def policy_paths() -> list[Path]:
    """Return the extracted policy markdown files in a stable order.

    Returns:
        Paths ending in ``.md``, sorted by file name.

    Raises:
        FileNotFoundError: The extraction directory is missing or empty.
    """
    if not DOCS.is_dir():
        raise FileNotFoundError(f"Extracted policies are not at {DOCS}")
    paths = sorted(DOCS.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown files in {DOCS}")
    return paths


def main() -> None:
    """Chunk the extracted policies and store them in both indexes."""
    paths = policy_paths()
    with KeywordIndex(str(KEYWORD_PATH)) as index:
        chunks = ingest(
            [str(path) for path in paths],
            index,
            get_collection(persist_directory=str(CHROMA_PATH)),
        )
    counts: Counter[tuple[str, str, str]] = Counter(
        (chunk.document_id, chunk.version or "none", chunk.status) for chunk in chunks
    )
    print(f"Ingested {len(paths)} files, {len(chunks)} chunks")
    for (document_id, version, status), count in sorted(counts.items()):
        print(f"{document_id} {version} {status}: {count}")


if __name__ == "__main__":
    main()
