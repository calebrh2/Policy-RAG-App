"""Provider-neutral contracts used by the RAG pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    metadata: Mapping[str, MetadataValue]


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


@runtime_checkable
class EmbeddingAdapter(Protocol):
    model_name: str
    dimensions: int
    max_input_tokens: int
    application_token_limit: int

    def count_tokens(self, text: str, *, is_query: bool = False) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class VectorStoreAdapter(Protocol):
    def upsert(
        self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]: ...


@runtime_checkable
class KeywordSearchAdapter(Protocol):
    def index(self, records: Sequence[VectorRecord]) -> None: ...

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]: ...


@runtime_checkable
class RerankerAdapter(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], *, limit: int
    ) -> list[SearchResult]: ...


@runtime_checkable
class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> str: ...


@runtime_checkable
class StructuredLLMAdapter(Protocol):
    model_name: str

    def generate_structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_schema: Mapping[str, Any],
    ) -> str: ...
