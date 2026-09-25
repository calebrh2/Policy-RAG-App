"""Session indexes for the evaluation harness.

Purpose
-------
Ingests each policy corpus once per pytest session into a temporary Chroma
database and keyword index. Tests score retrieval against that index.

Contents
--------
- ``EvalIndex``: one temporary collection, keyword database, and corpus name.
- ``eval_index``: session fixture that builds one index per corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.ChromaDB import get_collection
from rag.corpus import markdown_paths, resolve_corpus
from rag.ingest import ingest
from rag.keyword_index import KeywordIndex
from rag.retrieve import ChunkSearch


@dataclass(frozen=True)
class EvalIndex:
    """A temporary dense collection and keyword index over one policy corpus."""

    corpus: str
    collection: ChunkSearch
    keyword_path: str


@pytest.fixture(scope="session", params=("meridian", "coforge"))
def eval_index(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> EvalIndex:
    """Ingest one corpus into a temporary index.

    Args:
        request: Pytest request. ``param`` is ``meridian`` or ``coforge``.
        tmp_path_factory: Pytest factory for a session temporary directory.

    Returns:
        The corpus name, collection, and keyword database path.

    Raises:
        FileNotFoundError: That corpus directory has no markdown files.
    """
    corpus = resolve_corpus(str(request.param))
    paths = markdown_paths(corpus)
    root = tmp_path_factory.mktemp(f"eval-index-{corpus.name}")
    keyword_path = str(root / "chunks.sqlite")
    with KeywordIndex(keyword_path) as index:
        ingest(
            [str(path) for path in paths],
            index,
            get_collection(persist_directory=str(root / "chroma")),
        )
    return EvalIndex(
        corpus=corpus.name,
        collection=get_collection(persist_directory=str(root / "chroma")),
        keyword_path=keyword_path,
    )
