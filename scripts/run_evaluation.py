"""Run the 15 source-audited cases through the real RAG pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.config import get_settings
from rag.evaluation import load_evaluation_set, validate_relevant_chunks
from rag.evaluation_factory import build_pipeline
from rag.evaluation_harness import run_suite

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = get_settings()


def main() -> None:
    """Run the fixed question set through the live pipeline and save the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/evaluation/evaluation-set.jsonl"
    )
    parser.add_argument(
        "--rubrics", type=Path, default=ROOT / "data/evaluation/evaluation-rubrics.json"
    )
    parser.add_argument("--chunks", type=Path, default=SETTINGS.chunks_path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evaluation/runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    parser.add_argument("--model", default=SETTINGS.ollama_model)
    parser.add_argument("--ollama-url", default=SETTINGS.ollama_base_url)
    parser.add_argument("--top-k", type=int, default=SETTINGS.retrieval_top_k)
    parser.add_argument("--candidates", type=int, default=SETTINGS.retrieval_candidates)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidates < 1:
        parser.error("top-k and candidates must be positive")
    cases = load_evaluation_set(args.dataset)
    validate_relevant_chunks(cases, args.chunks)
    rubrics = json.loads(args.rubrics.read_text())
    with TemporaryDirectory(prefix="rag-evaluation-") as directory:
        pipeline = build_pipeline(Path(directory), args.chunks, args.model, args.ollama_url)
        rows = run_suite(
            pipeline,
            cases,
            rubrics,
            args.output_dir,
            top_k=args.top_k,
            candidates=args.candidates,
            inputs=[args.dataset, args.rubrics, args.chunks],
        )
    print(f"Results: {args.output_dir}")
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)


if __name__ == "__main__":
    main()
