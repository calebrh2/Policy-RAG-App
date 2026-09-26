"""Validated application models for grounded RAG answers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    """Source shown under an answer: document, version, section, pages, and chunk id."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_title: str
    version: str
    section_path: str
    page_start: int
    page_end: int


class AnswerClaim(BaseModel):
    """One independently verifiable statement and its supporting chunks."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1, description="One concise factual claim without citation markers"
    )
    cited_chunk_ids: list[str] = Field(
        min_length=1,
        description="Source chunk IDs that directly support this claim",
    )


class StructuredAnswerPayload(BaseModel):
    """Schema sent to the model through Ollama's format parameter."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Direct answer or an insufficient-evidence explanation",
    )
    sufficient_evidence: bool = Field(
        description=("True exactly when claims is non-empty; false only for a complete abstention")
    )
    claims: list[AnswerClaim] = Field(
        description="Supported factual claims; empty when evidence is insufficient"
    )

    @model_validator(mode="after")
    def derive_evidence_from_claims(self) -> StructuredAnswerPayload:
        """Use the cited claim list as the source of truth for evidence state.

        The boolean remains in the wire schema because it helps a model express an
        abstention. A non-empty claim list is stronger structured evidence than the
        redundant flag, so normalize disagreement instead of discarding a usable
        response from a smaller local model.
        """

        self.sufficient_evidence = bool(self.claims)
        return self


class GroundedAnswer(BaseModel):
    """Final answer text, whether the sources were enough, and the citations used."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sufficient_evidence: bool
    citations: tuple[Citation, ...]


class StructuredRoutePayload(BaseModel):
    """Validated output from the LLM routing fallback."""

    model_config = ConfigDict(extra="forbid")

    route: Literal["current", "historical", "comparison", "clarification"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
