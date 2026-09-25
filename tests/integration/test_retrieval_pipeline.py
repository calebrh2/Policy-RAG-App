"""Real golden retrieval checks, including the application's multipart context."""
from pathlib import Path

import pytest

from rag.evaluation import load_evaluation_set

CASES = load_evaluation_set(Path(__file__).resolve().parents[2] /
                            "data/evaluation/evaluation-set.jsonl")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_real_retrieval(case, live_evaluation_results):
    row = live_evaluation_results[case.id]
    assert row["error"] is None, row["error"]
    assert row["route_pass"], row
    if case.answerable:
        assert row["evidence_coverage"] == 1, row
