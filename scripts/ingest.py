"""Ingest one policy corpus into its Chroma collection and keyword index.

Run from the repository root, after ``uv sync --group dev --group rag``:

    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --corpus coforge

``--corpus`` selects the markdown directory and the index paths. It defaults
to ``RAG_CORPUS``, then ``meridian``. A new edition is stored beside the older
one. The first run downloads ``BAAI/bge-small-en-v1.5`` and embeds every chunk.
"""

from __future__ import annotations

import argparse
from collections import Counter

from rag.ChromaDB import get_collection
from rag.corpus import add_corpus_argument, markdown_paths, resolve_corpus
from rag.ingest import ingest
from rag.keyword_index import KeywordIndex


def main() -> None:
    """Chunk the selected corpus and store it in that corpus's indexes."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_corpus_argument(parser)
    corpus = resolve_corpus(parser.parse_args().corpus)
    paths = markdown_paths(corpus)
    with KeywordIndex(str(corpus.keyword)) as index:
        chunks = ingest(
            [str(path) for path in paths],
            index,
            get_collection(persist_directory=str(corpus.chroma)),
        )
    counts: Counter[tuple[str, str, str]] = Counter(
        (chunk.document_id, chunk.version or "none", chunk.status) for chunk in chunks
    )
    print(f"Ingested {corpus.name}: {len(paths)} files, {len(chunks)} chunks")
    for (document_id, version, status), count in sorted(counts.items()):
        print(f"{document_id} {version} {status}: {count}")


if __name__ == "__main__":
    main()
