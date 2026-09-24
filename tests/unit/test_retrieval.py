from rag.adapters.base import ChunkRecord, KeywordHit, SearchHit
from rag.retrieval import RRF_K, HybridRetriever, fuse


def test_chunk_found_by_both_indexes_ranks_above_a_single_hit() -> None:
    dense = [
        _dense("only-chroma", 0.1),
        _dense("both", 0.2),
    ]
    sparse = [
        _keyword("both", 3.0),
        _keyword("only-bm25", 2.0),
    ]

    hits = fuse(dense, sparse, limit=3)

    assert [hit.chunk_id for hit in hits] == ["both", "only-chroma", "only-bm25"]
    assert hits[0].score == (1 / (RRF_K + 2)) + (1 / (RRF_K + 1))
    assert hits[0].text == "both text 6,746.70"


def test_retriever_asks_both_indexes_for_the_current_version() -> None:
    store = _Store([_dense("table", 0.2)])
    keywords = _Keywords([_keyword("table", 4.0)])

    hits = HybridRetriever(store, keywords, candidates=20).retrieve(
        "What was India Scope 2 in 2023?",
        limit=5,
    )

    assert store.calls == [("What was India Scope 2 in 2023?", 20, "current")]
    assert keywords.calls == [("What was India Scope 2 in 2023?", 20, "current")]
    assert hits[0].chunk_id == "table"
    assert "6,746.70" in hits[0].text


def test_limit_keeps_only_the_requested_number_of_chunks() -> None:
    dense = [_dense("one", 0.1), _dense("two", 0.2), _dense("three", 0.3)]

    assert [hit.chunk_id for hit in fuse(dense, [], limit=2)] == ["one", "two"]


class _Store:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int, str | None]] = []

    def upsert(self, records: list[ChunkRecord]) -> None:
        raise AssertionError(records)

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[SearchHit]:
        self.calls.append((query, limit, version))
        return self._hits


class _Keywords:
    def __init__(self, hits: list[KeywordHit]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int, str | None]] = []

    def upsert(self, records: list[ChunkRecord]) -> None:
        raise AssertionError(records)

    def query(self, query: str, *, limit: int = 5, version: str | None = None) -> list[KeywordHit]:
        self.calls.append((query, limit, version))
        return self._hits


def _dense(chunk_id: str, distance: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version="current",
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text=f"{chunk_id} text 6,746.70",
        distance=distance,
    )


def _keyword(chunk_id: str, score: float) -> KeywordHit:
    return KeywordHit(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version="current",
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text=f"{chunk_id} text 6,746.70",
        score=score,
    )
