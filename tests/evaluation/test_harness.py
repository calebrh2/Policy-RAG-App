"""Retrieval recall against the fixed evaluation set.

Purpose
-------
Each question must retrieve the current chunk that states the answer.

Contents
--------
- ``gold_chunk_id``: the single current chunk a case names.
- ``retrieved_ids``: hybrid retrieval for one question.
- ``test_gold_chunks_are_unique``: the fixed set names one chunk each.
- ``test_retrieval_recall``: the gold chunk is in the retrieved list.
"""

from __future__ import annotations

from typing import Protocol

import pytest

from rag.keyword_index import KeywordIndex
from rag.retrieve import ChunkSearch, retrieve
from tests.evaluation.cases import CASES, EvalCase

_RETRIEVAL_LIMIT = 10


class _Index(Protocol):
    """The collection and keyword path the session fixture provides."""

    collection: ChunkSearch
    keyword_path: str


def gold_chunk_id(case: EvalCase, index: _Index) -> str:
    """Return the one current chunk the case names.

    Args:
        case: A question with a document, section, and needle.
        index: Ingested evaluation index.

    Returns:
        The matching chunk id.

    Raises:
        AssertionError: The needle matches none or more than one current chunk.
    """
    stored = index.collection.get(where={"searchable": True}, include=["documents", "metadatas"])
    ids = stored.get("ids")
    documents = stored.get("documents")
    metadatas = stored.get("metadatas")
    matched: list[str] = []
    if isinstance(ids, list) and isinstance(documents, list) and isinstance(metadatas, list):
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            if not isinstance(chunk_id, str) or not isinstance(text, str) or not isinstance(metadata, dict):
                continue
            if metadata.get("status") != "current":
                continue
            if metadata.get("document_id") != case.document_id:
                continue
            if metadata.get("section_path") != case.section_path:
                continue
            if case.needle not in text:
                continue
            matched.append(chunk_id)
    assert len(matched) == 1, f"{case.case_id} matched {matched}"
    return matched[0]


def retrieved_ids(case: EvalCase, index: _Index) -> list[str]:
    """Return hybrid-retrieval ids for one question, best first.

    Args:
        case: Evaluation question.
        index: Ingested evaluation index.

    Returns:
        Up to ten chunk ids.
    """
    with KeywordIndex(index.keyword_path) as keyword_index:
        hits = retrieve(case.question, index.collection, keyword_index, limit=_RETRIEVAL_LIMIT)
    return [hit.chunk_id for hit in hits]


def test_gold_chunks_are_unique(eval_index: _Index) -> None:
    """Each question names exactly one current chunk."""
    assert len(CASES) >= 8
    for case in CASES:
        gold_chunk_id(case, eval_index)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_retrieval_recall(case: EvalCase, eval_index: _Index) -> None:
    """The chunk that states the answer is among the retrieved hits.

    Args:
        case: One evaluation question.
        eval_index: Ingested evaluation index.
    """
    expected = gold_chunk_id(case, eval_index)
    found = retrieved_ids(case, eval_index)
    assert expected in found, f"{case.case_id} retrieved {found}"
