"""End-to-end query pipeline: route, retrieve, rerank, and generate."""

from __future__ import annotations

from rag.generation import GroundedAnswerGenerator
from rag.models import GroundedAnswer
from rag.retrieval import RetrievalService
from rag.router import QueryRoute, QueryRouter, RouteDecision


class RAGPipeline:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: GroundedAnswerGenerator,
        router: QueryRouter | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.router = router or QueryRouter()

    def answer(
        self,
        question: str,
        *,
        candidate_limit: int = 10,
        context_limit: int = 5,
        route: QueryRoute | None = None,
    ) -> tuple[GroundedAnswer, RouteDecision]:
        decision = (
            RouteDecision(route, "route explicitly selected by the caller")
            if route is not None
            else self.router.route(question)
        )
        results = self.retrieval.retrieve_statuses(
            question,
            statuses=decision.statuses,
            candidate_limit=candidate_limit,
            final_limit=context_limit,
        )
        return self.generator.generate(question, results), decision
