"""Contrast deliberately naive retrieval with version-aware retrieval."""

from __future__ import annotations

from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.ollama import OllamaAdapter
from rag.adapters.reranker import CrossEncoderRerankerAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.config import get_settings
from rag.indexing import load_searchable_records
from rag.retrieval import RetrievalService, reciprocal_rank_fusion

QUESTION = "Does the current Carbon Reduction Plan set a 15% reduction target for 2027?"


def main() -> None:
    """Show the outdated carbon plan answering a current-policy question, then the fix."""
    settings = get_settings()
    records = load_searchable_records(settings.chunks_path)
    embedder = BgeEmbeddingAdapter(
        model_name=settings.embedding_model,
        application_token_limit=settings.embedding_token_limit,
    )
    store = ChromaVectorStoreAdapter(
        settings.chroma_path,
        collection_name=settings.chroma_collection,
    )
    keyword = BM25KeywordSearchAdapter()
    keyword.index(records)
    reranker = CrossEncoderRerankerAdapter(model_name=settings.reranker_model)
    llm = OllamaAdapter(
        model_name=settings.ollama_model,
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        context_window=settings.ollama_context_window,
    )

    # Intentionally broken: retrieve every version, keep one result, and drop lineage.
    dense = store.search(embedder.embed_query(QUESTION), limit=10, filters=None)
    sparse = keyword.search(QUESTION, limit=10, filters=None)
    candidates = reciprocal_rank_fusion([dense, sparse], limit=20)
    naive_context = reranker.rerank(QUESTION, candidates, limit=1)

    print("INTENTIONALLY BROKEN: UNFILTERED TOP-1 RETRIEVAL")
    print(f"Question: {QUESTION}\n")
    for rank, result in enumerate(naive_context, 1):
        metadata = result.record.metadata
        print(
            f"{rank}. status={metadata['status']} version={metadata['version']} "
            f"section={metadata['section_path']}"
        )

    naive_passage = str(naive_context[0].record.metadata["original_text"])
    naive_answer = llm.generate(
        "Answer the question directly using only the passage.\n\n"
        f"QUESTION:\n{QUESTION}\n\nPASSAGE:\n{naive_passage}"
    )
    print(f"\nFlawed answer:\n{naive_answer}\n")

    service = RetrievalService(embedder, store, keyword, reranker)
    corrected_context = service.retrieve_statuses(
        QUESTION,
        statuses=("current",),
        candidate_limit=10,
        final_limit=1,
    )
    corrected_passages = "\n\n".join(
        f"CURRENT SOURCE {rank}:\n{result.record.metadata['original_text']}"
        for rank, result in enumerate(corrected_context, 1)
    )
    corrected_answer = llm.generate(
        "Answer using only current-policy evidence. If the question contains a false "
        "premise, say no and provide the actual current value.\n\n"
        f"QUESTION:\n{QUESTION}\n\nCURRENT POLICY SOURCES:\n{corrected_passages}"
    )

    print("CORRECTED: STATUS=CURRENT FILTER")
    print(f"Answer: {corrected_answer}")
    print("Sources:")
    for number, result in enumerate(corrected_context, 1):
        metadata = result.record.metadata
        print(
            f"[{number}] version={metadata['version']} section={metadata['section_path']} "
            f"chunk_id={result.record.id}"
        )

    print("\nDiagnosis:")
    print(
        "The corpus contains conflicting current and superseded policy versions. "
        "Unfiltered retrieval selected the superseded exact-match as its top result. "
        "Version/status metadata and current-version filtering are required to select "
        "the authoritative policy."
    )


if __name__ == "__main__":
    main()
