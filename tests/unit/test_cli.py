import json

from rag.cli import respond
from rag.retrieval import RetrievalHit


def test_respond_prints_the_answer_and_cited_chunk_text() -> None:
    payload = respond("What was India Scope 2 in 2023?", _Retriever(), _Reranker(), _Model())

    assert payload["answer"] == "India Scope 2 in 2023 was 6,746.70 tCO2e."
    citations = payload["citations"]
    assert isinstance(citations, list)
    assert citations[0]["section"] == "Current Year Emission: 2023"
    assert "6,746.70" in citations[0]["text"]
    json.dumps(payload)


class _Retriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        version: str | None = "current",
    ) -> list[RetrievalHit]:
        assert query
        assert limit == 20
        assert version == "current"
        return [_hit()]


class _Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        assert query
        return [1.0 for _ in texts]


class _Model:
    def complete(self, prompt: str) -> str:
        if "retrieval steps" in prompt:
            return json.dumps({"steps": ["What was India Scope 2 in 2023?"]})
        if "one word only" in prompt:
            return "current"
        assert "6,746.70" in prompt
        return json.dumps(
            {
                "text": "India Scope 2 in 2023 was 6,746.70 tCO2e.",
                "citations": [1],
            }
        )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="emissions",
        document_name="Carbon-Reduction-Plan",
        version="current",
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text="| Scope 2 | 22.33 | 6,746.70 |",
        score=0.03,
    )
