import json

from rag.cli import respond
from rag.planning import covers, decompose, focus_query
from rag.retrieval import RetrievalHit


def test_one_question_stays_one_step() -> None:
    plan = decompose("What was India Scope 2 in 2023?", _PlanModel(["What was India Scope 2 in 2023?"]))

    assert plan.steps == ["What was India Scope 2 in 2023?"]


def test_two_questions_become_two_steps() -> None:
    plan = decompose(
        "What is the carbon target, and when are single-use plastics eliminated?",
        _PlanModel(
            [
                "What is the carbon target?",
                "When are single-use plastics eliminated?",
            ]
        ),
    )

    assert plan.steps == [
        "What is the carbon target?",
        "When are single-use plastics eliminated?",
    ]


def test_a_bad_plan_stays_the_original_question() -> None:
    plan = decompose("What is the carbon target?", _RawModel("not json"))

    assert plan.steps == ["What is the carbon target?"]


def test_focus_query_drops_short_words() -> None:
    assert focus_query("by when must single-use plastics be eliminated") == "single plastics eliminated"


def test_coverage_needs_a_content_word_in_a_chunk() -> None:
    plastic = _hit("plastic", "single-use plastics eliminated by Dec 2026")
    carbon = _hit("carbon", "reduce them by 20% by 2030")

    assert covers("When are single-use plastics eliminated?", [plastic])
    assert not covers("When are single-use plastics eliminated?", [carbon])


def test_two_questions_retrieve_both_documents() -> None:
    payload = respond(
        "What is the carbon target, and when are single-use plastics eliminated?",
        _SplitRetriever(),
        _Reranker(),
        _AnswerModel(),
    )

    assert payload["answer"] == "The carbon target is 20% by 2030. Plastics are eliminated by Dec 2026."
    citations = payload["citations"]
    assert isinstance(citations, list)
    assert [item["document_name"] for item in citations] == [
        "Carbon-Reduction-Plan",
        "Single-use-Plastic-free-Policy",
    ]


def test_a_miss_retries_once_with_content_words() -> None:
    retriever = _RetryRetriever()

    payload = respond(
        "When are single-use plastics eliminated?",
        retriever,
        _Reranker(),
        _PlasticModel(),
    )

    assert retriever.queries == [
        "When are single-use plastics eliminated?",
        "single plastics eliminated",
    ]
    assert "Dec 2026" in str(payload["answer"])


class _PlanModel:
    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    def complete(self, prompt: str) -> str:
        assert "retrieval steps" in prompt
        return json.dumps({"steps": self._steps})


class _RawModel:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def complete(self, prompt: str) -> str:
        assert prompt
        return self._raw


class _AnswerModel:
    def complete(self, prompt: str) -> str:
        if "retrieval steps" in prompt:
            return json.dumps(
                {
                    "steps": [
                        "What is the carbon target?",
                        "When are single-use plastics eliminated?",
                    ]
                }
            )
        if "one word only" in prompt:
            return "current"
        assert "20%" in prompt
        assert "Dec 2026" in prompt
        return json.dumps(
            {
                "text": "The carbon target is 20% by 2030. Plastics are eliminated by Dec 2026.",
                "citations": [1, 2],
            }
        )


class _PlasticModel:
    def complete(self, prompt: str) -> str:
        if "retrieval steps" in prompt:
            return json.dumps({"steps": ["When are single-use plastics eliminated?"]})
        if "one word only" in prompt:
            return "current"
        assert "Dec 2026" in prompt
        return json.dumps(
            {
                "text": "Single-use plastics are eliminated by Dec 2026.",
                "citations": [1],
            }
        )


class _SplitRetriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        version: str | None = "current",
    ) -> list[RetrievalHit]:
        assert limit == 20
        assert version == "current"
        if "plastic" in query.casefold():
            return [_hit("plastic", "100% elimination of single-use plastics by Dec 2026", "Single-use-Plastic-free-Policy", "Targets")]
        return [_hit("carbon", "reduce them by 20% by 2030", "Carbon-Reduction-Plan", "Emissions reduction targets")]


class _RetryRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        version: str | None = "current",
    ) -> list[RetrievalHit]:
        assert limit == 20
        assert version == "current"
        self.queries.append(query)
        if query == "single plastics eliminated":
            return [_hit("plastic", "100% elimination of single-use plastics by Dec 2026", "Single-use-Plastic-free-Policy", "Targets")]
        return [_hit("carbon", "reduce them by 20% by 2030", "Carbon-Reduction-Plan", "Emissions reduction targets")]


class _Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        assert query
        return [1.0 for _ in texts]


def _hit(
    chunk_id: str,
    text: str,
    document_name: str = "Carbon-Reduction-Plan",
    section: str = "Targets",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_name=document_name,
        version="current",
        section=section,
        source_pages="1-1",
        text=text,
        score=0.03,
    )
