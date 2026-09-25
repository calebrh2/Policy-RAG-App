"""Evaluation dataset models and ground-truth validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QueryType = Literal[
    "semantic",
    "keyword",
    "exact",
    "numeric",
    "table",
    "historical",
    "comparison",
    "unanswerable",
]
ExpectedRoute = Literal["current", "historical", "comparison"]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    question: str = Field(min_length=5)
    query_type: QueryType
    expected_route: ExpectedRoute
    answerable: bool
    relevant_chunk_ids: list[str]
    expected_facts: list[str]
    notes: str = ""

    @model_validator(mode="after")
    def validate_ground_truth(self) -> EvaluationCase:
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError("answerable cases require relevant chunks")
        if self.answerable and not self.expected_facts:
            raise ValueError("answerable cases require expected facts")
        if not self.answerable and (self.relevant_chunk_ids or self.expected_facts):
            raise ValueError("unanswerable cases cannot define chunks or facts")
        return self


def load_evaluation_set(path: Path) -> list[EvaluationCase]:
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    return cases


def corpus_chunk_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["chunk_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def validate_relevant_chunks(cases: list[EvaluationCase], chunks_path: Path) -> None:
    available = corpus_chunk_ids(chunks_path)
    missing = {
        chunk_id
        for case in cases
        for chunk_id in case.relevant_chunk_ids
        if chunk_id not in available
    }
    if missing:
        raise ValueError(f"evaluation set references missing chunks: {sorted(missing)}")
