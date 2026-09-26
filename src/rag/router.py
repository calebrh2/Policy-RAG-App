"""Fast rule-based routing with a structured LLM fallback for ambiguity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import ValidationError

from rag.adapters.base import StructuredLLMAdapter
from rag.models import StructuredRoutePayload

COMPARISON_RE = re.compile(
    r"\b(compare|comparison|difference|differences|different|contrast|change|changed|changes|"
    r"versus|vs\.?|between|from .+ to|(?:old|previous|former|superseded) .+ current|current .+ (?:old|previous|former|superseded))\b",
    re.IGNORECASE,
)
HISTORICAL_RE = re.compile(
    r"\b(old|older|previous|prior|outdated|superseded|historical|former|"
    r"original|used to|2022|2027)\b",
    re.IGNORECASE,
)
CURRENT_RE = re.compile(
    r"\b(current|currently|latest|today|active|in force|now|present|applicable)\b",
    re.IGNORECASE,
)
AMBIGUOUS_TEMPORAL_RE = re.compile(
    r"\b((?:before|after) (?:the )?(?:revision|update|change)|revision|revised|"
    r"over time|at the time|back then|earlier version|since then|no longer|"
    r"still appl(?:y|ies)|evolved|moved)\b",
    re.IGNORECASE,
)


class QueryRoute(StrEnum):
    """Which policy version a question should search."""

    CLARIFICATION = "clarification"
    CURRENT = "current"
    HISTORICAL = "historical"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class RouteDecision:
    """The chosen route and a short reason a person can read."""

    route: QueryRoute
    reason: str

    @property
    def statuses(self) -> tuple[str, ...]:
        """Metadata status values this route is allowed to retrieve."""
        if self.route is QueryRoute.COMPARISON:
            return ("current", "superseded")
        if self.route is QueryRoute.HISTORICAL:
            return ("superseded",)
        if self.route is QueryRoute.CLARIFICATION:
            return ()
        return ("current",)


class Router(Protocol):
    """Chooses current, historical, comparison, or clarification for a question."""

    def route(self, query: str) -> RouteDecision:
        """Return the version intent for this question."""
        ...


class QueryRouter:
    """Resolve explicit version intent and default ordinary queries to current."""

    def explicit_route(self, query: str) -> RouteDecision | None:
        """Return a route when the wording clearly names a version, else None."""
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
        if CURRENT_RE.search(normalized):
            return RouteDecision(
                QueryRoute.CURRENT,
                "current-policy language requests the active version",
            )
        return None

    def needs_llm(self, query: str) -> bool:
        """True when version intent is ambiguous and the LLM fallback should run."""
        normalized = " ".join(query.split())
        return self.explicit_route(normalized) is None and bool(
            AMBIGUOUS_TEMPORAL_RE.search(normalized)
        )

    def route(self, query: str) -> RouteDecision:
        """Use an explicit version cue, otherwise treat the question as current."""
        explicit = self.explicit_route(query)
        if explicit is not None:
            return explicit
        return RouteDecision(
            QueryRoute.CURRENT,
            "ordinary policy questions default to the current version",
        )


ROUTER_SYSTEM_PROMPT = """Classify only the policy-version intent of the user query.
Use current for the active policy. Use historical when the user asks what applied before
a revision. Use comparison when the user asks how something moved, evolved, or changed
over time. Use clarification only when the version intent remains truly indeterminate.
Infer intent from meaning; exact keywords are unnecessary. Do not answer the policy
question. Give a brief reason and calibrated confidence."""


class HybridQueryRouter:
    """Apply deterministic rules first and call an LLM only for temporal ambiguity."""

    def __init__(
        self,
        llm: StructuredLLMAdapter,
        *,
        confidence_threshold: float = 0.70,
        rules: QueryRouter | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.rules = rules or QueryRouter()

    def route(self, query: str) -> RouteDecision:
        """Apply the rules first, then ask the model only for unclear time intent."""
        explicit = self.rules.explicit_route(query)
        if explicit is not None:
            return explicit
        if not self.rules.needs_llm(query):
            return self.rules.route(query)

        try:
            raw = self.llm.generate_structured(
                [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_schema=StructuredRoutePayload.model_json_schema(),
            )
            payload = StructuredRoutePayload.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            return RouteDecision(
                QueryRoute.CLARIFICATION,
                "ambiguous version intent and router output failed validation",
            )

        if payload.confidence < self.confidence_threshold:
            return RouteDecision(
                QueryRoute.CLARIFICATION,
                f"ambiguous version intent; router confidence {payload.confidence:.2f}",
            )
        return RouteDecision(
            QueryRoute(payload.route),
            f"structured LLM fallback: {payload.reason}",
        )


QUESTION_WORD = r"(?:(?:by|for|in|at|under)\s+)?(?:what|who|when|where|which|how|why|compare)"
QUESTION_PART_RE = re.compile(
    rf",\s*(?=(?:and\s+)?{QUESTION_WORD}\b)|"
    rf"\band\s+(?={QUESTION_WORD}\b)",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(r"^(?:what|who|when|where|which|how|why|compare)\b", re.IGNORECASE)


def decompose_question(query: str) -> tuple[str, ...]:
    """Split explicit multi-part questions while leaving ordinary prose intact."""
    parts = [part.strip(" ,") for part in QUESTION_PART_RE.split(query) if part.strip(" ,")]
    if len(parts) <= 1 or not QUESTION_START_RE.search(parts[0]):
        return (query.strip(),)
    return tuple(parts)


QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
NON_SPECIFIC_QUERY_WORDS = {
    "a",
    "an",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "explain",
    "how",
    "information",
    "is",
    "it",
    "me",
    "please",
    "policy",
    "question",
    "tell",
    "that",
    "the",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
}


def is_specific_question(query: str) -> bool:
    """Require at least one topic-bearing term before retrieval."""
    tokens = (token.casefold() for token in QUERY_TOKEN_RE.findall(query))
    return any(token not in NON_SPECIFIC_QUERY_WORDS for token in tokens)
