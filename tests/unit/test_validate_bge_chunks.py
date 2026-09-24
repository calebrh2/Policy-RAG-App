from __future__ import annotations

import pytest

from rag.chunk_validation import validate_chunks


class FakeAdapter:
    application_token_limit = 480

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_validate_chunks_attaches_real_count() -> None:
    chunks = [
        {"chunk_id": "one", "searchable": True, "retrieval_text": "three token input"}
    ]

    result = validate_chunks(chunks, FakeAdapter())

    assert result[0]["embedding_token_count"] == 3


def test_validate_chunks_rejects_oversized_searchable_input() -> None:
    chunks = [
        {"chunk_id": "too-long", "searchable": True, "retrieval_text": "x " * 481}
    ]

    with pytest.raises(ValueError, match="structurally split"):
        validate_chunks(chunks, FakeAdapter())
