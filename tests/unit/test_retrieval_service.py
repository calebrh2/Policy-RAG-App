from __future__ import annotations

from rag.adapters.base import SearchResult, VectorRecord
from rag.retrieval import RetrievalService


class Embedder:
    model_name = "fake"
    dimensions = 2
    max_input_tokens = 512
    application_token_limit = 480

    def count_tokens(self, text: str, *, is_query: bool = False) -> int:
        return 1

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class CapturingSearch:
    def __init__(self) -> None:
        self.filters = None

    def search(self, query, *, limit, filters=None):  # type: ignore[no-untyped-def]
        self.filters = filters
        return [SearchResult(VectorRecord("id", "text", filters or {}), 1.0)]

    def upsert(self, records, embeddings):  # type: ignore[no-untyped-def]
        pass

    def index(self, records):  # type: ignore[no-untyped-def]
        pass


def test_dense_and_sparse_default_to_current_version() -> None:
    dense = CapturingSearch()
    sparse = CapturingSearch()
    service = RetrievalService(Embedder(), dense, sparse)  # type: ignore[arg-type]

    service.dense("target")
    service.sparse("target")

    assert dense.filters == {"status": "current"}
    assert sparse.filters == {"status": "current"}


def test_explicit_status_can_request_superseded_version() -> None:
    dense = CapturingSearch()
    service = RetrievalService(Embedder(), dense, CapturingSearch())  # type: ignore[arg-type]

    service.dense("old target", filters={"status": "superseded"})

    assert dense.filters == {"status": "superseded"}
