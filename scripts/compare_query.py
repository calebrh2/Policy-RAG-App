"""Log vector-only and hybrid rankings for one query.

Run from the repository root, after that corpus has been ingested:

    uv run python scripts/compare_query.py "travel expense report deadline POL-EXP-200"

Vector-only is the dense Chroma ranking. Hybrid is reciprocal-rank fusion of
that ranking and BM25. The reranker and the chat model are not called.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_retrieval import TOP_K, hybrid_ids, vector_only_ids

from rag.ChromaDB import get_collection
from rag.corpus import add_corpus_argument, resolve_corpus
from rag.keyword_index import KeywordIndex
from rag.retrieve import ChunkSearch

log = logging.getLogger("compare")


def chunk_labels(collection: ChunkSearch) -> dict[str, str]:
    """Map chunk ids to document title and section.

    Args:
        collection: Dense store that supports ``get``.

    Returns:
        One label per stored chunk id.
    """
    stored = collection.get(include=["metadatas"])
    labels: dict[str, str] = {}
    ids = stored.get("ids")
    metadatas = stored.get("metadatas")
    if not isinstance(ids, list) or not isinstance(metadatas, list):
        return labels
    for chunk_id, metadata in zip(ids, metadatas, strict=True):
        if not isinstance(chunk_id, str) or not isinstance(metadata, dict):
            continue
        title = metadata.get("document_title") or metadata.get("document_id")
        labels[chunk_id] = f"{title} / {metadata.get('section_path')}"
    return labels


def log_ranking(heading: str, ids: list[str], labels: dict[str, str]) -> None:
    """Log one ranked list.

    Args:
        heading: Name of the retrieval method.
        ids: Ranked chunk ids, best first.
        labels: Display text for each chunk id.
    """
    log.info("%s", heading)
    for rank, chunk_id in enumerate(ids, start=1):
        log.info("  %s %s", rank, labels.get(chunk_id, chunk_id))


def main() -> None:
    """Log both rankings for the query passed on the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    add_corpus_argument(parser)
    parser.add_argument("query", help="Question to rank with both methods.")
    args = parser.parse_args()
    corpus = resolve_corpus(args.corpus)
    log.info("corpus=%s k=%s", corpus.name, TOP_K)
    log.info("query=%s", args.query)
    log.info("Loading dense collection")
    collection = get_collection(persist_directory=str(corpus.chroma))
    labels = chunk_labels(collection)
    vector_ids = vector_only_ids(args.query, collection, TOP_K)
    log.info("Loading keyword index")
    with KeywordIndex(str(corpus.keyword)) as index:
        fused_ids = hybrid_ids(args.query, collection, index, TOP_K)
    log_ranking("vector-only", vector_ids, labels)
    log_ranking("hybrid", fused_ids, labels)


if __name__ == "__main__":
    main()
