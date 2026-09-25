from __future__ import annotations

import pytest

from rag.router import QueryRoute, QueryRouter, decompose_question, is_specific_question


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the emissions reduction target?", QueryRoute.CURRENT),
        ("What was the old emissions target?", QueryRoute.HISTORICAL),
        ("What target was used in 2022?", QueryRoute.HISTORICAL),
        ("Compare the old and current targets", QueryRoute.COMPARISON),
        ("How did the target change?", QueryRoute.COMPARISON),
    ],
)
def test_query_router(query: str, expected: QueryRoute) -> None:
    assert QueryRouter().route(query).route is expected


def test_routes_map_to_expected_statuses() -> None:
    router = QueryRouter()
    assert router.route("current policy").statuses == ("current",)
    assert router.route("previous policy").statuses == ("superseded",)
    assert router.route("compare versions").statuses == ("current", "superseded")

def test_decomposes_explicit_multi_part_question() -> None:
    parts = decompose_question(
        "What is the reduction target, what is the net-zero year, "
        "and what were UK emissions in 2023?"
    )
    assert parts == (
        "What is the reduction target",
        "what is the net-zero year",
        "what were UK emissions in 2023?",
    )


def test_does_not_split_ordinary_conjunctions() -> None:
    assert decompose_question("What are the water and emissions targets?") == (
        "What are the water and emissions targets?",
    )

@pytest.mark.parametrize("query", ["What", "Why?", "Tell me", "What is it?"])
def test_rejects_non_specific_queries(query: str) -> None:
    assert is_specific_question(query) is False


@pytest.mark.parametrize(
    "query", ["What is WEEE?", "Water?", "What is the emissions target?"]
)
def test_accepts_topic_bearing_queries(query: str) -> None:
    assert is_specific_question(query) is True

class FakeStructuredLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[object, object]] = []

    def generate_structured(self, messages, *, response_schema):
        self.calls.append((messages, response_schema))
        return self.response


def test_hybrid_router_skips_llm_for_obvious_and_ordinary_queries() -> None:
    from rag.router import HybridQueryRouter

    llm = FakeStructuredLLM('{"route":"historical","confidence":1,"reason":"x"}')
    router = HybridQueryRouter(llm)

    assert router.route("What is the current target?").route is QueryRoute.CURRENT
    assert router.route("What is the water policy scope?").route is QueryRoute.CURRENT
    assert router.route("What was the previous target?").route is QueryRoute.HISTORICAL
    assert llm.calls == []


def test_hybrid_router_uses_validated_llm_fallback_for_ambiguous_time_intent() -> None:
    from rag.router import HybridQueryRouter

    llm = FakeStructuredLLM(
        '{"route":"comparison","confidence":0.91,"reason":"asks how it moved"}'
    )
    decision = HybridQueryRouter(llm).route("Has our carbon ambition moved over time?")

    assert decision.route is QueryRoute.COMPARISON
    assert len(llm.calls) == 1
    messages, schema = llm.calls[0]
    assert "properties" in schema
    assert '"properties"' not in str(messages)


def test_hybrid_router_requests_clarification_for_low_confidence() -> None:
    from rag.router import HybridQueryRouter

    llm = FakeStructuredLLM(
        '{"route":"historical","confidence":0.40,"reason":"uncertain"}'
    )
    decision = HybridQueryRouter(llm).route("What applied before the revision?")

    assert decision.route is QueryRoute.CLARIFICATION


def test_hybrid_router_rejects_invalid_structured_output() -> None:
    from rag.router import HybridQueryRouter

    decision = HybridQueryRouter(FakeStructuredLLM('{"route":"past"}')).route(
        "What applied before the revision?"
    )
    assert decision.route is QueryRoute.CLARIFICATION
