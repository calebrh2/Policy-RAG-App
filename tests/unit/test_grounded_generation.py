from __future__ import annotations

import json

import pytest

from rag.adapters.base import SearchResult, VectorRecord
from rag.generation import GenerationValidationError, GroundedAnswerGenerator


class FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.messages = None
        self.schema = None

    def generate_structured(self, messages, *, response_schema):  # type: ignore[no-untyped-def]
        self.messages = messages
        self.schema = response_schema
        return json.dumps(self.payload)


def source() -> SearchResult:
    return SearchResult(
        VectorRecord(
            "chunk-1",
            "retrieval text",
            {
                "document_title": "Carbon Plan",
                "version": "2024",
                "status": "current",
                "section_path": "Targets",
                "page_start": 2,
                "page_end": 2,
                "original_text": "The target is 20 percent by 2030.",
            },
        ),
        1.0,
    )


def test_generation_builds_citations_from_trusted_metadata() -> None:
    llm = FakeLLM(
        {
            "answer": "The target is 20 percent by 2030.",
            "sufficient_evidence": True,
            "cited_chunk_ids": ["chunk-1", "chunk-1"],
        }
    )
    answer = GroundedAnswerGenerator(llm).generate("What is the target?", [source()])  # type: ignore[arg-type]

    assert answer.sufficient_evidence is True
    assert len(answer.citations) == 1
    assert answer.citations[0].document_title == "Carbon Plan"
    assert "The target is 20 percent" in llm.messages[1]["content"]
    assert "properties" not in llm.messages[1]["content"]


def test_generation_rejects_unknown_citation() -> None:
    llm = FakeLLM(
        {
            "answer": "Unsupported",
            "sufficient_evidence": True,
            "cited_chunk_ids": ["invented"],
        }
    )

    with pytest.raises(GenerationValidationError, match="not supplied"):
        GroundedAnswerGenerator(llm).generate("Question?", [source()])  # type: ignore[arg-type]


def test_no_results_returns_insufficient_without_calling_llm() -> None:
    llm = FakeLLM({})
    answer = GroundedAnswerGenerator(llm).generate("Question?", [])  # type: ignore[arg-type]
    assert answer.sufficient_evidence is False
    assert answer.citations == ()
    assert llm.messages is None
