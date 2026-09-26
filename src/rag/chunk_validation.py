"""Embedding-model validation for generated retrieval chunks."""

from __future__ import annotations

from typing import Any, Protocol


class TokenCounter(Protocol):
    """Anything that can count tokens the way the embedding model does."""

    application_token_limit: int

    def count_tokens(self, text: str) -> int:
        """Return the token count for one chunk's retrieval text."""
        ...


def validate_chunks(
    chunks: list[dict[str, Any]], adapter: TokenCounter
) -> list[dict[str, Any]]:
    """Attach exact token counts and reject searchable inputs that would be too long."""
    oversized: list[tuple[str, int]] = []
    validated: list[dict[str, Any]] = []

    for chunk in chunks:
        count = adapter.count_tokens(str(chunk["retrieval_text"]))
        item = dict(chunk)
        item["embedding_token_count"] = count
        validated.append(item)
        if bool(chunk["searchable"]) and count > adapter.application_token_limit:
            oversized.append((str(chunk["chunk_id"]), count))

    if oversized:
        details = "\n".join(f"- {chunk_id}: {count}" for chunk_id, count in oversized)
        raise ValueError(
            "Searchable chunks exceed the BGE application limit and must be "
            f"structurally split before indexing:\n{details}"
        )
    return validated
