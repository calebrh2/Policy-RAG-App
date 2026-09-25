from __future__ import annotations

from pathlib import Path

from rag.evaluation import load_evaluation_set, validate_relevant_chunks

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_set_is_valid_and_covers_required_cases() -> None:
    cases = load_evaluation_set(ROOT / "data/evaluation/evaluation-set.jsonl")
    validate_relevant_chunks(cases, ROOT / "data/chunks/chunks.jsonl")

    assert len(cases) >= 8
    assert {case.expected_route for case in cases} == {
        "current",
        "historical",
        "comparison",
    }
    assert any(case.query_type == "table" for case in cases)
    assert any(case.query_type == "unanswerable" for case in cases)
    assert any(not case.answerable for case in cases)
