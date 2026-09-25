"""Retrieval recall and recorded generation quality for the fixed set.

Purpose
-------
Each question names one current chunk. Retrieval recall is the share of those
questions whose chunk comes back. The newest generation run must also clear
the key-information bar. Both bars are 0.9. Retrieval has 9 questions, so 8
of 9 is 0.889 and fails. Generation has 11, so 10 of 11 is 0.909 and passes.

Contents
--------
- ``gold_chunk_id``: the single current chunk a case names.
- ``retrieved_ids``: hybrid retrieval for one question.
- ``test_gold_chunks_are_unique``: the fixed set names one chunk each.
- ``test_retrieval_recall``: recall across the set is at least 0.9.
- ``test_latest_generation_run_meets_key_information_threshold``: the newest
  generation run's key-information mean is at least 0.9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from rag.keyword_index import KeywordIndex
from rag.retrieve import ChunkSearch, retrieve
from tests.evaluation.cases import CASES, EvalCase

_RETRIEVAL_LIMIT = 10
_SCORE_THRESHOLD = 0.9
_GENERATION_RUNS = Path(__file__).resolve().parents[2] / "runs" / "generation"


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


def test_retrieval_recall(eval_index: _Index) -> None:
    """The share of questions that retrieve the gold chunk is at least 0.9.

    Args:
        eval_index: Ingested evaluation index.

    Raises:
        AssertionError: Recall is below 0.9. The message names the misses.
    """
    misses = [
        case.case_id
        for case in CASES
        if gold_chunk_id(case, eval_index) not in retrieved_ids(case, eval_index)
    ]
    recall = (len(CASES) - len(misses)) / len(CASES)
    assert recall >= _SCORE_THRESHOLD, f"recall={recall:.3f} misses={misses}"


def test_latest_generation_run_meets_key_information_threshold() -> None:
    """The newest generation run's key-information mean is at least 0.9.

    Raises:
        AssertionError: No run file exists, or the mean is below 0.9.
    """
    runs = sorted(_GENERATION_RUNS.glob("generation-*.json"))
    assert runs, f"No generation run in {_GENERATION_RUNS}. Run scripts/evaluate_generation.py."
    latest = runs[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    score = payload["means"]["key_information"]
    assert isinstance(score, (int, float)) and not isinstance(score, bool)
    assert score >= _SCORE_THRESHOLD, (
        f"{latest.name} key_information={score:.3f} is below {_SCORE_THRESHOLD}"
    )
