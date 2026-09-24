"""Deterministic query routing for policy-version retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

COMPARISON_RE = re.compile(
    r"\b(compare|comparison|difference|different|change|changed|changes|"
    r"versus|vs\.?|between|from .+ to|old .+ current|current .+ old)\b",
    re.IGNORECASE,
)
HISTORICAL_RE = re.compile(
    r"\b(old|older|previous|prior|outdated|superseded|historical|former|"
    r"original|used to|was the|were the|2022|2027)\b",
    re.IGNORECASE,
)


class QueryRoute(StrEnum):
    CURRENT = "current"
    HISTORICAL = "historical"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    reason: str

    @property
    def statuses(self) -> tuple[str, ...]:
        if self.route is QueryRoute.COMPARISON:
            return ("current", "superseded")
        if self.route is QueryRoute.HISTORICAL:
            return ("superseded",)
        return ("current",)


class QueryRouter:
    """Prefer current policy unless the question explicitly asks otherwise."""

    def route(self, query: str) -> RouteDecision:
        normalized = " ".join(query.split())
        if COMPARISON_RE.search(normalized):
            return RouteDecision(
                QueryRoute.COMPARISON,
                "comparison language requests current and superseded versions",
            )
        if HISTORICAL_RE.search(normalized):
            return RouteDecision(
                QueryRoute.HISTORICAL,
                "historical language requests the superseded version",
            )
        return RouteDecision(
            QueryRoute.CURRENT,
            "ordinary policy questions default to the current version",
        )
