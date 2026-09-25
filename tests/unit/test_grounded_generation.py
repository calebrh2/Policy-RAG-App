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


class SequencedFakeLLM:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = iter(payloads)
        self.calls = 0
        self.messages = []

    def generate_structured(self, messages, *, response_schema):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.messages.append(messages)
        return next(self.payloads)


def source(chunk_id: str = "chunk-1", *, section: str = "Targets") -> SearchResult:
    return SearchResult(
        VectorRecord(
            chunk_id,
            "retrieval text",
            {
                "document_title": "Carbon Plan",
                "version": "2024",
                "status": "current",
                "section_path": section,
                "page_start": 2,
                "page_end": 2,
                "original_text": "The target is 20 percent by 2030.",
            },
        ),
        1.0,
    )


def approximate_source() -> SearchResult:
    result = source()
    return SearchResult(
        VectorRecord(
            result.record.id,
            "Reach 10% integration by 2025 and ~50% integration by 2030.",
            result.record.metadata,
        ),
        result.score,
    )


def test_generation_renders_inline_citations_from_trusted_metadata() -> None:
    llm = FakeLLM(
        {
            "answer": "The target is 20 percent by 2030.",
            "sufficient_evidence": True,
            "claims": [
                {
                    "text": "The target is 20 percent by 2030.",
                    "cited_chunk_ids": ["chunk-1", "chunk-1"],
                }
            ],
        }
    )
    answer = GroundedAnswerGenerator(llm).generate(
        "What is the target?", [source()]
    )  # type: ignore[arg-type]

    assert answer.answer == "The target is 20 percent by 2030 [1]."
    assert answer.sufficient_evidence is True
    assert len(answer.citations) == 1
    assert answer.citations[0].document_title == "Carbon Plan"
    assert "retrieval text" in llm.messages[1]["content"]
    assert "properties" not in llm.messages[1]["content"]


def test_generation_numbers_sources_by_first_claim_use() -> None:
    llm = FakeLLM(
        {
            "answer": "Two supported claims.",
            "sufficient_evidence": True,
            "claims": [
                {"text": "The target is 20 percent.", "cited_chunk_ids": ["chunk-2"]},
                {"text": "The deadline is 2030.", "cited_chunk_ids": ["chunk-1"]},
            ],
        }
    )
    answer = GroundedAnswerGenerator(llm).generate(
        "What is the target and deadline?",
        [source("chunk-1"), source("chunk-2", section="Reduction")],
    )  # type: ignore[arg-type]

    assert answer.answer == "The target is 20 percent [1]. The deadline is 2030 [2]."
    assert [citation.chunk_id for citation in answer.citations] == ["chunk-2", "chunk-1"]


def test_generation_rejects_unknown_citation() -> None:
    llm = FakeLLM(
        {
            "answer": "Unsupported",
            "sufficient_evidence": True,
            "claims": [{"text": "Unsupported", "cited_chunk_ids": ["invented"]}],
        }
    )

    with pytest.raises(GenerationValidationError, match="after 2 attempt"):
        GroundedAnswerGenerator(llm).generate("Question?", [source()])  # type: ignore[arg-type]


def test_generation_retries_invalid_structured_output_once() -> None:
    valid = json.dumps(
        {
            "answer": "The target is 20 percent by 2030.",
            "sufficient_evidence": True,
            "claims": [
                {
                    "text": "The target is 20 percent by 2030.",
                    "cited_chunk_ids": ["chunk-1"],
                }
            ],
        }
    )
    llm = SequencedFakeLLM(["not json", valid])

    answer = GroundedAnswerGenerator(llm).generate(
        "What is the target?", [source()]
    )  # type: ignore[arg-type]

    assert answer.sufficient_evidence is True
    assert llm.calls == 2
    assert "could not be accepted" in llm.messages[1][-1]["content"]


def test_generation_normalizes_inconsistent_evidence_boolean() -> None:
    llm = FakeLLM(
        {
            "answer": "The target is 20 percent by 2030.",
            "sufficient_evidence": False,
            "claims": [
                {
                    "text": "The target is 20 percent by 2030.",
                    "cited_chunk_ids": ["chunk-1"],
                }
            ],
        }
    )

    answer = GroundedAnswerGenerator(llm).generate(
        "What is the target?", [source()]
    )  # type: ignore[arg-type]

    assert answer.sufficient_evidence is True
    assert answer.answer == "The target is 20 percent by 2030 [1]."


def test_no_results_returns_insufficient_without_calling_llm() -> None:
    llm = FakeLLM({})
    answer = GroundedAnswerGenerator(llm).generate("Question?", [])  # type: ignore[arg-type]
    assert answer.sufficient_evidence is False
    assert answer.citations == ()
    assert llm.messages is None


def test_source_approximation_is_not_rendered_as_an_exact_percentage() -> None:
    payload = {
        "answer": "The targets are 10% and 50%.",
        "sufficient_evidence": True,
        "claims": [
            {
                "text": "The targets are 10% by 2025 and 50% by 2030.",
                "cited_chunk_ids": ["chunk-1"],
            }
        ],
    }
    llm = FakeLLM(payload)

    answer = GroundedAnswerGenerator(llm).generate(
        "What are the targets?", [approximate_source()]
    )  # type: ignore[arg-type]

    assert answer.answer == "The targets are 10% by 2025 and approximately 50% by 2030 [1]."
