"""Build a real, isolated evaluation pipeline."""

from pathlib import Path

from rag.adapters.embeddings import BgeEmbeddingAdapter
from rag.adapters.keyword_search import BM25KeywordSearchAdapter
from rag.adapters.ollama import OllamaAdapter
from rag.adapters.reranker import CrossEncoderRerankerAdapter
from rag.adapters.vector_store import ChromaVectorStoreAdapter
from rag.config import get_settings
from rag.generation import GroundedAnswerGenerator
from rag.indexing import index_records, load_searchable_records
from rag.pipeline import RAGPipeline
from rag.retrieval import RetrievalService
from rag.router import HybridQueryRouter


def build_pipeline(index_path: Path, chunks: Path, model: str, url: str) -> RAGPipeline:
    """Index the chunks in a fresh Chroma directory and return a live pipeline."""
    settings = get_settings()
    records = load_searchable_records(chunks)
    embedder = BgeEmbeddingAdapter(
        model_name=settings.embedding_model,
        application_token_limit=settings.embedding_token_limit,
    )
    store = ChromaVectorStoreAdapter(index_path, collection_name="evaluation_chunks")
    index_records(records, embedder, store)
    keyword = BM25KeywordSearchAdapter()
    keyword.index(records)
    llm = OllamaAdapter(
        model_name=model,
        base_url=url,
        timeout_seconds=settings.ollama_timeout_seconds,
        context_window=settings.ollama_context_window,
        think=settings.ollama_think,
    )
    return RAGPipeline(
        RetrievalService(
            embedder,
            store,
            keyword,
            CrossEncoderRerankerAdapter(model_name=settings.reranker_model),
        ),
        GroundedAnswerGenerator(llm),
        router=HybridQueryRouter(llm, confidence_threshold=settings.router_confidence),
    )
