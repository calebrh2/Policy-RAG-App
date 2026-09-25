"""End-to-end query pipeline: route, retrieve, rerank, and generate."""

from __future__ import annotations

from collections.abc import Callable

from rag.adapters.base import SearchResult
from rag.generation import GroundedAnswerGenerator
from rag.models import GroundedAnswer
from rag.retrieval import RetrievalService
from rag.router import (
    QueryRoute,
    QueryRouter,
    RouteDecision,
    Router,
    decompose_question,
    is_specific_question,
)


class RAGPipeline:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: GroundedAnswerGenerator,
        router: Router | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.router = router or QueryRouter()

    def answer(
        self,
        question: str,
        *,
        candidate_limit: int = 10,
        on_context: Callable[[list[SearchResult]], None] | None = None,
        context_limit: int = 3,
        route: QueryRoute | None = None,
    ) -> tuple[GroundedAnswer, RouteDecision]:
        if not is_specific_question(question):
            return (
                GroundedAnswer(
                    answer=(
                        "Please ask a specific question about a policy, target, date, "
                        "rule, responsibility, or document."
                    ),
                    sufficient_evidence=False,
                    citations=(),
                ),
                RouteDecision(
                    QueryRoute.CLARIFICATION,
                    "query was too vague to retrieve policy evidence",
                ),
            )
        decision = (
            RouteDecision(route, "route explicitly selected by the caller")
            if route is not None
            else self.router.route(question)
        )
        if decision.route is QueryRoute.CLARIFICATION:
            return (
                GroundedAnswer(
                    answer=(
                        "Please clarify whether you want the current policy, the "
                        "superseded policy, or a comparison of both versions."
                    ),
                    sufficient_evidence=False,
                    citations=(),
                ),
                decision,
            )
        parts = decompose_question(question)
        if len(parts) == 1:
            results = self.retrieval.retrieve_statuses(
                question,
                statuses=decision.statuses,
                candidate_limit=candidate_limit,
                final_limit=context_limit,
            )
        else:
            per_part_limit = max(1, context_limit // len(parts))
            selected = [
                result
                for part in parts
                for result in self.retrieval.retrieve_statuses(
                    part,
                    statuses=decision.statuses,
                    candidate_limit=candidate_limit,
                    final_limit=per_part_limit,
                )
            ]
            results = list({result.record.id: result for result in selected}.values())
        if on_context is not None:
            on_context(results)
        return self.generator.generate(question, results), decision
