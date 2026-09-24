from rag.adapters.reranker import CrossEncoderReranker
from rag.retrieval import RetrievalHit, rerank


def test_rerank_puts_the_higher_cross_encoder_score_first() -> None:
    scorer = _FakeScorer({"plastic": 0.1, "6,746.70": 4.2})
    hits = rerank(
        "What was India Scope 2 in 2023?",
        [
            _hit("plastic", "Eliminate single-use plastic by Dec 2026"),
            _hit("emissions", "India Scope 2 in 2023 was 6,746.70"),
        ],
        CrossEncoderReranker(scorer=scorer),
    )

    assert [hit.chunk_id for hit in hits] == ["emissions", "plastic"]
    assert hits[0].score == 4.2
    assert scorer.pairs[0][0] == "What was India Scope 2 in 2023?"
    assert "6,746.70" in scorer.pairs[1][1]


def test_empty_hit_list_does_not_call_the_model() -> None:
    scorer = _FakeScorer({})

    assert rerank("India Scope 2", [], CrossEncoderReranker(scorer=scorer)) == []
    assert scorer.pairs == []


def test_limit_keeps_the_top_cross_encoder_scores() -> None:
    scorer = _FakeScorer({"one": 1.0, "two": 3.0, "three": 2.0})
    hits = rerank(
        "question",
        [_hit("one", "one"), _hit("two", "two"), _hit("three", "three")],
        CrossEncoderReranker(scorer=scorer),
        limit=2,
    )

    assert [hit.chunk_id for hit in hits] == ["two", "three"]


class _FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.pairs: list[list[str]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [
            next(score for needle, score in self._scores.items() if needle in text)
            for _query, text in pairs
        ]


def _hit(chunk_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version="current",
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text=text,
        score=0.01,
    )
