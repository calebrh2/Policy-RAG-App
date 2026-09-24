from __future__ import annotations

import pytest

from rag.router import QueryRoute, QueryRouter


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
