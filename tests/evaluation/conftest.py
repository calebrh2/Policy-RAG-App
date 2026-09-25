"""Session index for the evaluation harness.

Purpose
-------
Ingests the extracted policies once per pytest session into a temporary
Chroma database and keyword index. Tests score retrieval against that index.

Contents
--------
- ``EvalIndex``: the temporary collection and keyword database.
- ``eval_index``: session fixture that builds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rag.ChromaDB import get_collection
from rag.ingest import ingest
from rag.keyword_index import KeywordIndex
from rag.retrieve import ChunkSearch

_DOCS = Path(__file__).resolve().parents[2] / "data" / "extracted" / "RAG-documents"


@dataclass(frozen=True)
class EvalIndex:
    """A temporary dense collection and keyword index over the policy corpus."""

    collection: ChunkSearch
    keyword_path: str


@pytest.fixture(scope="session")
def eval_index(tmp_path_factory: pytest.TempPathFactory) -> EvalIndex:
    """Ingest the extracted policies into a temporary index.

    Args:
        tmp_path_factory: Pytest factory for a session temporary directory.

    Returns:
        The collection and keyword database path.

    Raises:
        FileNotFoundError: The extracted policy directory has no markdown files.
    """
    paths = sorted(_DOCS.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown files in {_DOCS}")
    root = tmp_path_factory.mktemp("eval-index")
    keyword_path = str(root / "chunks.sqlite")
    with KeywordIndex(keyword_path) as index:
        ingest([str(path) for path in paths], index, get_collection(persist_directory=str(root / "chroma")))
    return EvalIndex(
        collection=get_collection(persist_directory=str(root / "chroma")),
        keyword_path=keyword_path,
    )
