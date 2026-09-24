from typing import Any

import pytest

from rag.adapters.base import EmbeddingAdapter
from rag.adapters.embeddings import (
    BgeEmbeddingAdapter,
    InputTooLongError,
    build_embedding_adapter,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        assert add_special_tokens is True
        assert truncation is False
        self.inputs.append(text)
        return [101, *range(len(text.split())), 102]


class FakeModel:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def encode(self, value: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((value, kwargs))
        if isinstance(value, list):
            return [[1, 2, 3] for _ in value]
        return [1, 2, 3]


def test_bge_adapter_keeps_documents_plain_and_instructs_queries() -> None:
    model = FakeModel()
    adapter = BgeEmbeddingAdapter(model=model)

    documents = adapter.embed_documents(["policy content"])
    query = adapter.embed_query("current target")

    assert documents == [[1.0, 2.0, 3.0]]
    assert query == [1.0, 2.0, 3.0]
    assert model.calls[0][0] == ["policy content"]
    assert model.calls[1][0].startswith(BgeEmbeddingAdapter.QUERY_INSTRUCTION)
    assert all(call[1]["normalize_embeddings"] is True for call in model.calls)


def test_bge_adapter_rejects_document_before_silent_truncation() -> None:
    adapter = BgeEmbeddingAdapter(model=FakeModel(), application_token_limit=5)

    with pytest.raises(InputTooLongError, match="Split the input"):
        adapter.embed_documents(["one two three four"])


def test_factory_returns_protocol_compatible_bge_adapter() -> None:
    adapter = build_embedding_adapter("bge", model=FakeModel())

    assert isinstance(adapter, BgeEmbeddingAdapter)
    assert isinstance(adapter, EmbeddingAdapter)
    assert adapter.dimensions == 384
    assert adapter.max_input_tokens == 512
