import json

from rag.adapters.ollama import ModelRequestError, OllamaClient, TransientHttpError
from rag.generation import GenerationError, generate
from rag.retrieval import RetrievalHit


def test_answer_copies_the_figure_from_the_cited_chunk() -> None:
    model = _Model(
        json.dumps(
            {
                "text": "India Scope 2 in 2023 was 6,746.70 tCO2e.",
                "citations": [1],
            }
        )
    )

    answer = generate("What was India Scope 2 in 2023?", [_hit()], model)

    assert answer.text == "India Scope 2 in 2023 was 6,746.70 tCO2e."
    assert answer.citations[0].section == "Current Year Emission: 2023"
    assert "6,746.70" in model.prompt


def test_rejects_a_number_that_is_not_in_the_cited_chunk() -> None:
    model = _Model(
        json.dumps(
            {
                "text": "India Scope 2 in 2023 was 9,999.99 tCO2e.",
                "citations": [1],
            }
        )
    )

    try:
        generate("What was India Scope 2 in 2023?", [_hit()], model)
    except GenerationError as exc:
        assert "9,999.99" in str(exc)
    else:
        raise AssertionError("expected the invented figure to be rejected")


def test_rejects_a_citation_that_was_not_retrieved() -> None:
    model = _Model(json.dumps({"text": "Water is shared.", "citations": [2]}))

    try:
        generate("What is the water purpose?", [_hit()], model)
    except GenerationError as exc:
        assert "not one of the retrieved chunks" in str(exc)
    else:
        raise AssertionError("expected the unknown citation to be rejected")


def test_no_chunks_does_not_call_the_model() -> None:
    model = _Model("{}")

    answer = generate("What was India Scope 2 in 2023?", [], model)

    assert answer.citations == []
    assert model.prompt == ""


def test_http_failure_is_retried_and_a_rejected_call_is_not() -> None:
    flaky = _FlakyPoster()
    client = OllamaClient("http://localhost:11434/v1", "mistral", poster=flaky)

    assert client.complete("question") == "ok"
    assert flaky.calls == 3

    rejected = _RejectPoster()
    client = OllamaClient("http://localhost:11434/v1", "mistral", poster=rejected)
    try:
        client.complete("question")
    except ModelRequestError:
        assert rejected.calls == 1
    else:
        raise AssertionError("expected the rejected call to stop")


class _Model:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return self._reply


class _FlakyPoster:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url: str, body: bytes) -> bytes:
        self.calls += 1
        if self.calls < 3:
            raise TransientHttpError("down")
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


class _RejectPoster:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url: str, body: bytes) -> bytes:
        self.calls += 1
        raise ModelRequestError("HTTP 400")


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="emissions",
        document_name="Carbon-Reduction-Plan",
        version="current",
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text="| Scope 2 | 22.33 | 6,746.70 |",
        score=4.2,
    )


