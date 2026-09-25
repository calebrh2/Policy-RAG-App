"""Real generated-answer checks; no canned answers or label-fed prompts."""
from pathlib import Path

import pytest

from rag.evaluation import load_evaluation_set

CASES = load_evaluation_set(Path(__file__).resolve().parents[2] /
                            "data/evaluation/evaluation-set.jsonl")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_real_answer(case, live_evaluation_results):
    row = live_evaluation_results[case.id]
    assert row["error"] is None, row["error"]
    assert row["answer_pass"], (
        f"{case.id}: fact checks={row.get('fact_checks')} answer={row.get('answer')}"
    )
    assert row["citation_ids_valid"], row
    if case.answerable:
        assert row["citation_evidence_coverage"] == 1, row
