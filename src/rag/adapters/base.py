"""Provider-neutral contracts used by the RAG pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class VectorRecord:
    """One stored chunk: its id, the text that was embedded, and source metadata."""

    id: str
    text: str
    metadata: Mapping[str, MetadataValue]


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk plus the score from the search that returned it."""

    record: VectorRecord
    score: float


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Turns document chunks and questions into vectors."""

    model_name: str
    dimensions: int
    max_input_tokens: int
    application_token_limit: int

    def count_tokens(self, text: str, *, is_query: bool = False) -> int:
        """Return how many tokens this text uses, including any query instruction."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk text with no query instruction prepended."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a question so it can be compared with stored document vectors."""
        ...


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """Stores embeddings and returns the nearest chunks for a query vector."""

    def upsert(
        self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Insert or replace chunks and their embeddings."""
        ...

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        """Return the closest stored chunks, optionally limited by metadata."""
        ...


@runtime_checkable
class KeywordSearchAdapter(Protocol):
    """Finds chunks by exact words rather than vector similarity."""

    def index(self, records: Sequence[VectorRecord]) -> None:
        """Build the keyword index from these chunks."""
        ...

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[SearchResult]:
        """Return chunks whose words best match the query."""
        ...


@runtime_checkable
class RerankerAdapter(Protocol):
    """Rescores an initial candidate list for one question."""

    def rerank(
        self, query: str, candidates: Sequence[SearchResult], *, limit: int
    ) -> list[SearchResult]:
        """Return the candidates most relevant to the question, best first."""
        ...


@runtime_checkable
class LLMAdapter(Protocol):
    """Generates free-text from one prompt."""

    def generate(self, prompt: str) -> str:
        """Return the model's text reply."""
        ...


@runtime_checkable
class StructuredLLMAdapter(Protocol):
    """Generates JSON that must match a supplied schema."""

    model_name: str

    def generate_structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_schema: Mapping[str, Any],
    ) -> str:
        """Return a JSON string constrained to the response schema."""
        ...
