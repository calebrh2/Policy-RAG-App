"""Ask a policy question and print a JSON answer with cited chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from rag.adapters.ollama import LanguageModel
from rag.adapters.reranker import Reranker
from rag.generation import GenerationError, generate
from rag.models import Citation
from rag.retrieval import RetrievalHit, rerank
from rag.routing import comparison_hits, route

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
    """Retrieve, rerank, and return the answer with the cited chunk text."""
    hits = _hits(question, retriever, reranker, model)
    try:
        answer = generate(question, hits, model)
    except GenerationError as exc:
        return {"question": question, "error": str(exc), "answer": "", "citations": []}
    return {
        "question": question,
        "answer": answer.text,
        "citations": [_citation(citation, hits) for citation in answer.citations],
    }


def _hits(
    question: str,
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
) -> list[RetrievalHit]:
    chosen = route(question, model)
    if chosen == "compare":
        current = retriever.retrieve(question, limit=_CANDIDATES, version="current")
        outdated = retriever.retrieve(question, limit=_CANDIDATES, version="outdated")
        if not outdated:
            return rerank(question, current, reranker, limit=_ANSWER_CHUNKS)
        return comparison_hits(current, outdated, limit=_ANSWER_CHUNKS)
    version = "outdated" if chosen == "outdated" else "current"
    fused = retriever.retrieve(question, limit=_CANDIDATES, version=version)
    return rerank(question, fused, reranker, limit=_ANSWER_CHUNKS)


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
