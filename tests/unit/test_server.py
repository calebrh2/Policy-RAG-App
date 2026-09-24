from rag.retrieval import RetrievalHit
from rag.server import ask_payload


def test_ask_rejects_a_blank_question() -> None:
    status, payload = ask_payload({"question": "  "}, _Retriever(), _Reranker(), _Model())

    assert status == 400
    assert payload["error"] == "Question is required."


def test_ask_returns_the_cli_answer() -> None:
    status, payload = ask_payload(
        {"question": "What was India Scope 2 in 2023?"},
        _Retriever(),
        _Reranker(),
        _Model(),
    )

    assert status == 200
    assert payload["answer"] == "India Scope 2 in 2023 was 6,746.70 tCO2e."


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
        return [
            RetrievalHit(
                chunk_id="emissions",
                document_name="Carbon-Reduction-Plan",
                version="current",
                section="Current Year Emission: 2023",
                source_pages="2-2",
                text="| Scope 2 | 22.33 | 6,746.70 |",
                score=0.03,
            )
        ]


class _Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        assert query
        return [1.0 for _ in texts]


class _Model:
    def complete(self, prompt: str) -> str:
        if "one word only" in prompt:
            return "current"
        return '{"text": "India Scope 2 in 2023 was 6,746.70 tCO2e.", "citations": [1]}'
