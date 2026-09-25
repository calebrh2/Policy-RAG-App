"""Reproducible evaluation of the actual application pipeline.

Only questions enter the pipeline. Labels are used afterwards for scoring.
Lexical answer checks and citation coverage are proxies, not entailment judgments.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Any

from rag.adapters.base import SearchResult
from rag.evaluation import EvaluationCase
from rag.pipeline import RAGPipeline


def coverage(groups: list[list[str]], ids: Sequence[str]) -> float | None:
    if not groups:
        return None
    found = set(ids)
    return sum(bool(found.intersection(group)) for group in groups) / len(groups)


def check_answer(text: str, checks: list[list[str]]) -> list[bool]:
    """Each fact has AND requirements; a regex may express accepted alternatives."""
    normalized = re.sub(r"\s+", " ", text.casefold().replace(",", ""))
    return [
        all(re.search(pattern, normalized, re.IGNORECASE) is not None for pattern in fact)
        for fact in checks
    ]


def evaluate_case(
    pipeline: RAGPipeline,
    case: EvaluationCase,
    rubric: dict[str, Any],
    *,
    top_k: int = 5,
    candidates: int = 10,
) -> dict[str, Any]:
    context: list[SearchResult] = []
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": case.id,
        "question": case.question,
        "expected_route": case.expected_route,
        "expected_answerable": case.answerable,
        "expected_facts": case.expected_facts,
        "error": None,
    }
    try:
        answer, decision = pipeline.answer(
            case.question,
            candidate_limit=candidates,
            context_limit=top_k,
            on_context=context.extend,
        )
        ids = [result.record.id for result in context]
        cited = [citation.chunk_id for citation in answer.citations]
        groups = rubric["evidence_groups"]
        facts = check_answer(answer.answer, rubric["fact_checks"])
        forbidden = check_answer(answer.answer, [[p] for p in rubric["forbidden_patterns"]])
        if case.answerable:
            answer_pass = answer.sufficient_evidence and all(facts) and not any(forbidden)
        else:
            refusal = check_answer(answer.answer, [rubric["abstention_patterns"]])[0]
            answer_pass = not answer.sufficient_evidence and refusal and not any(forbidden)
        relevant = set(case.relevant_chunk_ids)
        row.update(
            actual_route=decision.route.value,
            route_reason=decision.reason,
            route_pass=decision.route.value == case.expected_route,
            answer=answer.answer,
            sufficient_evidence=answer.sufficient_evidence,
            fact_checks=facts,
            fact_coverage=mean(facts) if facts else None,
            forbidden_match=any(forbidden),
            answer_pass=bool(answer_pass),
            retrieved_chunk_ids=ids,
            cited_chunk_ids=cited,
            context=[
                {
                    "id": r.record.id,
                    "text": r.record.text,
                    "metadata": dict(r.record.metadata),
                    "score": r.score,
                }
                for r in context
            ],
            evidence_coverage=coverage(groups, ids),
            citation_evidence_coverage=coverage(groups, cited),
            citation_ids_valid=set(cited).issubset(ids),
            # Combined multipart context is not a single ranked list.
            context_recall=(len(relevant.intersection(ids)) / len(relevant) if relevant else None),
        )
        from rag.router import decompose_question

        row["subquestions"] = list(decompose_question(case.question))
        row["ranking_metrics_applicable"] = len(row["subquestions"]) == 1
        if relevant and row["ranking_metrics_applicable"]:
            for k in (1, 3, 5):
                row[f"recall_at_{k}"] = len(relevant.intersection(ids[:k])) / len(relevant)
                row[f"hit_at_{k}"] = bool(relevant.intersection(ids[:k]))
            row["reciprocal_rank"] = next(
                (1 / rank for rank, id in enumerate(ids, 1) if id in relevant), 0.0
            )
        row["passed"] = (
            row["route_pass"]
            and row["answer_pass"]
            and row["citation_ids_valid"]
            and (
                not case.answerable
                or (row["evidence_coverage"] == 1 and row["citation_evidence_coverage"] == 1)
            )
        )
    except Exception as error:  # noqa: BLE001 - record per-case failures and continue the suite
        row.update(
            error=f"{type(error).__name__}: {error}",
            passed=False,
            context=[{"id": r.record.id, "text": r.record.text} for r in context],
        )
    row["latency_seconds"] = time.perf_counter() - started
    return row


def validate_rubrics(cases: list[EvaluationCase], rubrics: dict[str, Any]) -> None:
    for case in cases:
        rubric = rubrics[case.id]
        groups = rubric["evidence_groups"]
        if case.answerable:
            assert groups and all(groups), f"{case.id}: missing evidence groups"
            assert set().union(*map(set, groups)) == set(case.relevant_chunk_ids)
            assert len(rubric["fact_checks"]) == len(case.expected_facts)
        else:
            assert not groups and not rubric["fact_checks"]
            assert rubric["abstention_patterns"]
        for fact in rubric["fact_checks"]:
            assert fact
        patterns = [p for f in rubric["fact_checks"] for p in f]
        patterns += rubric["forbidden_patterns"] + rubric["abstention_patterns"]
        for pattern in patterns:
            re.compile(pattern)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    return ordered[max(0, ceil(probability * len(ordered)) - 1)]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, answer, citation, routing, and latency metrics."""

    if not rows:
        raise ValueError("cannot summarize an empty evaluation run")
    total = len(rows)
    answerable = [row for row in rows if row["expected_answerable"]]
    ranked = [
        row
        for row in answerable
        if row.get("ranking_metrics_applicable") and row.get("error") is None
    ]
    latencies = [float(row["latency_seconds"]) for row in rows]
    summary: dict[str, Any] = {
        "cases": total,
        "errors": sum(row.get("error") is not None for row in rows),
        "route_accuracy": mean(bool(row.get("route_pass")) for row in rows),
        "answer_accuracy": mean(bool(row.get("answer_pass")) for row in rows),
        "full_pass_rate": mean(bool(row.get("passed")) for row in rows),
        "mean_context_recall": mean(float(row.get("context_recall") or 0) for row in answerable),
        "mean_evidence_coverage": mean(
            float(row.get("evidence_coverage") or 0) for row in answerable
        ),
        "mean_citation_evidence_coverage": mean(
            float(row.get("citation_evidence_coverage") or 0) for row in answerable
        ),
        "mean_latency_seconds": mean(latencies),
        "p50_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "manual_claim_review_required": True,
    }
    for k in (1, 3, 5):
        summary[f"mean_recall_at_{k}"] = mean(float(row.get(f"recall_at_{k}", 0)) for row in ranked)
        summary[f"hit_rate_at_{k}"] = mean(bool(row.get(f"hit_at_{k}")) for row in ranked)
    summary["mrr"] = mean(float(row.get("reciprocal_rank", 0)) for row in ranked)
    return summary


def run_suite(
    pipeline: RAGPipeline,
    cases: list[EvaluationCase],
    rubrics: dict[str, Any],
    output: Path,
    *,
    top_k: int = 5,
    candidates: int = 10,
    inputs: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    validate_rubrics(cases, rubrics)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "top_k": top_k,
        "candidates": candidates,
        "generator": pipeline.generator.llm.model_name,
        "router": type(pipeline.router).__name__,
        "embedder": pipeline.retrieval.embedder.model_name,
        "review_status": "AI source-audited; human approval pending",
        "metric_limits": "Lexical facts and evidence-ID coverage; no automated claim entailment.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (output / "results.jsonl").open("w") as handle:
        for case in cases:
            row = evaluate_case(
                pipeline, case, rubrics[case.id], top_k=top_k, candidates=candidates
            )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{case.id}: {'PASS' if row['passed'] else 'CHECK'} "
                f"({row['latency_seconds']:.1f}s)",
                flush=True,
            )
    summary = summarize_rows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nEVALUATION SUMMARY", flush=True)
    for name, value in summary.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{name}: {rendered}", flush=True)
    review = [
        "# Evaluation review\n\nAutomatic checks are proxies. Review each claim and citation.\n"
    ]
    for row in rows:
        review.append(
            f"\n## {row['id']} — {'PASS' if row['passed'] else 'CHECK'}\n\n"
            f"Question: {row['question']}\n\n"
            f"Expected facts: {row['expected_facts']}\n\n"
            f"Answer: {row.get('answer', row['error'])}\n\n"
            f"Fact checks: {row.get('fact_checks')}\n\n"
            f"Citation coverage: {row.get('citation_evidence_coverage')}\n"
        )
    (output / "review.md").write_text("".join(review))
    return rows
