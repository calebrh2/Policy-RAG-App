"""Named policy corpora and their document and index paths.

Purpose
-------
Maps ``meridian`` and ``coforge`` to a markdown directory and separate Chroma
and keyword indexes. Scripts and the API select one corpus so the two
collections do not overwrite each other.

Contents
--------
- ``Corpus``: one corpus's paths.
- ``resolve_corpus``: ``--corpus``, then ``RAG_CORPUS``, then ``meridian``.
- ``markdown_paths``: the ``.md`` files in a corpus directory.
- ``add_corpus_argument``: the shared ``--corpus`` flag.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = "meridian"
_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Corpus:
    """Documents and indexes for one policy set."""

    name: str
    documents: Path
    chroma: Path
    keyword: Path


CORPORA: dict[str, Corpus] = {
    "meridian": Corpus(
        name="meridian",
        documents=_ROOT / "data/extracted/RAG-documents",
        chroma=_ROOT / "data/chromadb/meridian",
        keyword=_ROOT / "data/keyword/meridian.sqlite",
    ),
    "coforge": Corpus(
        name="coforge",
        documents=_ROOT / "data/extracted/previous",
        chroma=_ROOT / "data/chromadb/coforge",
        keyword=_ROOT / "data/keyword/coforge.sqlite",
    ),
}


def resolve_corpus(name: str | None = None) -> Corpus:
    """Return the corpus named by the argument, ``RAG_CORPUS``, or the default.

    Args:
        name: Explicit corpus name. ``None`` or an empty string reads
            ``RAG_CORPUS``, then ``meridian``.

    Returns:
        The matching corpus.

    Raises:
        ValueError: The name is not a known corpus.
    """
    selected = name or os.environ.get("RAG_CORPUS") or DEFAULT_CORPUS
    selected = selected.strip()
    corpus = CORPORA.get(selected)
    if corpus is None:
        known = ", ".join(CORPORA)
        raise ValueError(f"Unknown corpus {selected!r}. Known corpora: {known}")
    return corpus


def markdown_paths(corpus: Corpus) -> list[Path]:
    """Return the markdown files in a corpus directory, in file-name order.

    Args:
        corpus: Corpus whose ``documents`` directory is read. Subdirectories
            are not included.

    Returns:
        Paths ending in ``.md``.

    Raises:
        FileNotFoundError: The directory is missing or has no markdown files.
    """
    if not corpus.documents.is_dir():
        raise FileNotFoundError(f"Extracted policies are not at {corpus.documents}")
    paths = sorted(corpus.documents.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown files in {corpus.documents}")
    return paths


def add_corpus_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--corpus`` to a pipeline script.

    Args:
        parser: Parser for one script. The default is unset so
            ``resolve_corpus`` can still read ``RAG_CORPUS``.
    """
    parser.add_argument(
        "--corpus",
        choices=tuple(CORPORA),
        default=None,
        help="Policy corpus. Defaults to RAG_CORPUS, then meridian.",
    )
