"""Offline regression tests for scoring logic, not real-model accuracy claims."""

import json
from pathlib import Path

from rag.evaluation import load_evaluation_set
from rag.evaluation_harness import (
    check_answer,
    coverage,
    evaluate_case,
    summarize_rows,
    validate_rubrics,
)
from rag.models import GroundedAnswer
from rag.pipeline import RAGPipeline
from rag.router import QueryRoute, RouteDecision

ROOT = Path(__file__).resolve().parents[2]


def test_complete_rubrics():
    cases = load_evaluation_set(ROOT / "data/evaluation/evaluation-set.jsonl")
    assert 15 <= len(cases) <= 20
    rubrics = json.loads((ROOT / "data/evaluation/evaluation-rubrics.json").read_text())
    validate_rubrics(cases, rubrics)


def test_evidence_alternatives_are_or_and_required_groups_are_and():
    assert coverage([["a", "b"]], ["b"]) == 1
    assert coverage([["a", "b"], ["c"]], ["b"]) == 0.5
    assert coverage([["a", "b"], ["c"]], ["b", "c"]) == 1
    assert coverage([], []) is None


def test_numeric_pair_checks_reject_swapped_targets():
    checks = [[r"10\s*%.{0,100}2025"], [r"50\s*%.{0,100}2030"]]
    assert check_answer("10% by 2025; 50% by 2030.", checks) == [True, True]
    assert check_answer("10% by 2030; 50% by 2025.", checks) != [True, True]


class RecordingRetrieval:
    def __init__(self):
        self.questions = []

    def retrieve_statuses(self, question, **kwargs):
        self.questions.append(question)
        return []


class RecordingGenerator:
    def generate(self, question, results):
        return GroundedAnswer(
            answer="Insufficient evidence", sufficient_evidence=False, citations=()
        )


def test_multipart_calls_actual_pipeline_and_records_subquestions():
    retrieval = RecordingRetrieval()
    pipeline = RAGPipeline(retrieval, RecordingGenerator())
    pipeline.answer("What is the carbon target, and what is the water target?")
    assert len(retrieval.questions) == 2


def test_vague_query_does_not_retrieve():
    retrieval = RecordingRetrieval()
    _, decision = RAGPipeline(retrieval, RecordingGenerator()).answer("What")
    assert decision.route == QueryRoute.CLARIFICATION
    assert retrieval.questions == []


def test_clarification_does_not_retrieve():
    class Clarify:
        def route(self, question):
            return RouteDecision(QueryRoute.CLARIFICATION, "unclear version")

    retrieval = RecordingRetrieval()
    _, decision = RAGPipeline(retrieval, RecordingGenerator(), Clarify()).answer("policy revision")
    assert decision.route == QueryRoute.CLARIFICATION
    assert retrieval.questions == []


def test_backend_error_is_recorded_as_failure():
    class Broken:
        def answer(self, *args, **kwargs):
            raise RuntimeError("Ollama unavailable")

    case = load_evaluation_set(ROOT / "data/evaluation/evaluation-set.jsonl")[0]
    row = evaluate_case(Broken(), case, {})
    assert not row["passed"]
    assert "Ollama unavailable" in row["error"]


def test_summary_aggregates_retrieval_answer_and_latency_metrics():
    rows = [
        {
            "expected_answerable": True,
            "ranking_metrics_applicable": True,
            "error": None,
            "route_pass": True,
            "answer_pass": True,
            "passed": True,
            "context_recall": 1.0,
            "evidence_coverage": 1.0,
            "citation_evidence_coverage": 1.0,
            "latency_seconds": 1.0,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "hit_at_1": True,
            "hit_at_3": True,
            "hit_at_5": True,
            "reciprocal_rank": 1.0,
        },
        {
            "expected_answerable": True,
            "ranking_metrics_applicable": True,
            "error": None,
            "route_pass": True,
            "answer_pass": False,
            "passed": False,
            "context_recall": 0.5,
            "evidence_coverage": 0.5,
            "citation_evidence_coverage": 0.0,
            "latency_seconds": 3.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.5,
            "recall_at_5": 0.5,
            "hit_at_1": False,
            "hit_at_3": True,
            "hit_at_5": True,
            "reciprocal_rank": 1 / 3,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["answer_accuracy"] == 0.5
    assert summary["mean_recall_at_1"] == 0.5
    assert summary["mean_recall_at_3"] == 0.75
    assert summary["hit_rate_at_3"] == 1.0
    assert summary["mrr"] == 2 / 3
    assert summary["mean_latency_seconds"] == 2.0
    assert summary["p50_latency_seconds"] == 2.0
    assert summary["p95_latency_seconds"] == 3.0
