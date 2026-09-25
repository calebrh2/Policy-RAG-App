"""Ask a policy question and print a JSON answer with cited chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from rag.adapters.ollama import LanguageModel
from rag.adapters.reranker import Reranker
from rag.generation import GenerationError, generate
from rag.models import Citation
from rag.planning import covers, decompose, focus_query
from rag.retrieval import RetrievalHit, rerank
from rag.routing import (
    comparison_hits,
    resolve_route,
    should_compare_steps,
    version_terms,
    versioned_hits,
)

_CANDIDATES = 20
_ANSWER_CHUNKS = 5


class _Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        version: str | None = "current",
    ) -> list[RetrievalHit]:
        """Return fused chunks for a question."""


def respond(
    question: str,
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
) -> dict[str, object]:
    """Retrieve each part of the question, then return the answer with cited chunk text."""
    hits = _planned_hits(question, retriever, reranker, model)
    try:
        answer = generate(question, hits, model)
    except GenerationError as exc:
        return {"question": question, "error": str(exc), "answer": "", "citations": []}
    return {
        "question": question,
        "answer": answer.text,
        "citations": [_citation(citation, hits) for citation in answer.citations],
    }


def _planned_hits(
    question: str,
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
) -> list[RetrievalHit]:
    multi_version, single_version = _version_terms()
    chosen = resolve_route(
        question,
        model,
        multi_version=multi_version,
        single_version=single_version,
    )
    if chosen == "compare":
        return _search(question, "compare", retriever, reranker, multi_version=multi_version)
    plan = decompose(question, model)
    labels = [
        resolve_route(step, model, multi_version=multi_version, single_version=single_version)
        for step in plan.steps
    ]
    if should_compare_steps(labels, plan.steps, single_version=single_version):
        return _search(question, "compare", retriever, reranker, multi_version=multi_version)
    groups = [
        _step(step, retriever, reranker, model, multi_version, single_version)
        for step in plan.steps
    ]
    if len(groups) == 1:
        return groups[0]
    return _union(groups)


def _step(
    question: str,
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
    multi_version: frozenset[str],
    single_version: frozenset[str],
) -> list[RetrievalHit]:
    chosen = resolve_route(
        question,
        model,
        multi_version=multi_version,
        single_version=single_version,
    )
    hits = _search(question, chosen, retriever, reranker)
    if covers(question, hits):
        return hits
    focused = focus_query(question)
    if not focused or focused == question.casefold():
        return hits
    return _search(focused, "current", retriever, reranker, rank_query=question)


def _search(
    question: str,
    chosen: str,
    retriever: _Retriever,
    reranker: Reranker,
    *,
    rank_query: str | None = None,
    multi_version: frozenset[str] = frozenset(),
) -> list[RetrievalHit]:
    query = question
    rank = rank_query or question
    if chosen == "compare":
        current = versioned_hits(
            retriever.retrieve(query, limit=_CANDIDATES, version="current"),
            multi_version,
        )
        outdated = versioned_hits(
            retriever.retrieve(query, limit=_CANDIDATES, version="outdated"),
            multi_version,
        )
        if not outdated:
            return rerank(rank, current, reranker, limit=_ANSWER_CHUNKS)
        return comparison_hits(current, outdated, limit=_ANSWER_CHUNKS)
    version = "outdated" if chosen == "outdated" else "current"
    fused = retriever.retrieve(query, limit=_CANDIDATES, version=version)
    return rerank(rank, fused, reranker, limit=_ANSWER_CHUNKS)


def _version_terms() -> tuple[frozenset[str], frozenset[str]]:
    path = Path("data/chunks/chunks.jsonl")
    if not path.is_file():
        return frozenset(), frozenset()
    records = [
        (str(raw["document_name"]), str(raw["version"]))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for raw in [json.loads(line)]
    ]
    return version_terms(records)


def _union(groups: list[list[RetrievalHit]]) -> list[RetrievalHit]:
    seen: set[str] = set()
    merged: list[RetrievalHit] = []
    for hits in groups:
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            merged.append(hit)
    return merged


def main() -> None:
    retriever, reranker, model = _stack()
    while True:
        try:
            question = input("Question: ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(json.dumps(respond(question, retriever, reranker, model), indent=2))


def _stack() -> tuple[_Retriever, Reranker, LanguageModel]:
    from rag.adapters.embeddings import SentenceTransformerEmbedder
    from rag.adapters.keyword_search import ChromaBm25Index
    from rag.adapters.ollama import OllamaClient
    from rag.adapters.reranker import CrossEncoderReranker
    from rag.adapters.vector_store import ChromaVectorStore
    from rag.retrieval import HybridRetriever

    keywords = ChromaBm25Index()
    keywords.upsert_jsonl(Path("data/chunks/chunks.jsonl"))
    store = ChromaVectorStore(
        SentenceTransformerEmbedder(),
        persist_directory=Path("data/chromadb"),
    )
    return HybridRetriever(store, keywords), CrossEncoderReranker(), OllamaClient.from_env()


def _citation(citation: Citation, hits: list[RetrievalHit]) -> dict[str, str]:
    text = next(
        (
            hit.text
            for hit in hits
            if hit.document_name == citation.document_name
            and hit.section == citation.section
            and hit.source_pages == citation.source_pages
        ),
        "",
    )
    return {
        "document_name": citation.document_name,
        "section": citation.section,
        "source_pages": citation.source_pages,
        "text": text,
    }


if __name__ == "__main__":
    main()
