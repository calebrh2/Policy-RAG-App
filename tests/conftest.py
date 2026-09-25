"""Live evaluations are explicit; backend failures fail requested live runs."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.evaluation import load_evaluation_set, validate_relevant_chunks
from rag.evaluation_factory import build_pipeline
from rag.evaluation_harness import run_suite

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--run-rag-evals",
        action="store_true",
        help="Run real BGE/Chroma/cross-encoder/Ollama golden evaluations",
    )


@pytest.fixture(scope="session")
def live_evaluation_results(request, tmp_path_factory):
    if not request.config.getoption("--run-rag-evals"):
        pytest.skip("Use --run-rag-evals to run real model evaluation")
    dataset = ROOT / "data/evaluation/evaluation-set.jsonl"
    chunks = ROOT / "data/chunks/chunks.jsonl"
    rubric_path = ROOT / "data/evaluation/evaluation-rubrics.json"
    cases = load_evaluation_set(dataset)
    validate_relevant_chunks(cases, chunks)
    pipeline = build_pipeline(
        tmp_path_factory.mktemp("rag-index"),
        chunks,
        os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )
    output = ROOT / "data/evaluation/runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    rows = run_suite(
        pipeline,
        cases,
        json.loads(rubric_path.read_text()),
        output,
        inputs=[dataset, chunks, rubric_path],
    )
    return {r["id"]: r for r in rows}
