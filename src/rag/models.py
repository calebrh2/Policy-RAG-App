"""Validated application models for grounded RAG answers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_title: str
    version: str
    section_path: str
    page_start: int
    page_end: int


class StructuredAnswerPayload(BaseModel):
    """Schema sent to the model through Ollama's format parameter."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sufficient_evidence: bool
    cited_chunk_ids: list[str]

    @model_validator(mode="after")
    def evidence_requires_citations(self) -> StructuredAnswerPayload:
        if self.sufficient_evidence and not self.cited_chunk_ids:
            raise ValueError("sufficient answers require at least one cited chunk")
        return self


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sufficient_evidence: bool
    citations: tuple[Citation, ...]
