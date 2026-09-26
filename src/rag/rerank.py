"""Rerank hybrid-retrieval candidates with a cross-encoder.

Purpose
-------
Scores the fused chunks from ``retrieve`` and keeps the top matches. The
default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2``. Any ``Reranker``
can be passed instead. This does not rebuild the dense or keyword indexes.

Contents
--------
- ``RERANK_MODEL``: the default cross-encoder id.
- ``default_reranker``: load that model on CPU.
- ``rerank``: score candidates and return the top chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from rag.adapters.reranker import CrossEncoderReranker, Reranker
from rag.retrieve import RetrievedChunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def default_reranker() -> CrossEncoderReranker:
    """Return the project's default cross-encoder.

    Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on CPU. The first call
    downloads the model weights.

    Returns:
        A ready ``CrossEncoderReranker``.
    """
    return CrossEncoderReranker(model_name=RERANK_MODEL, device="cpu")


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    reranker: Reranker,
    *,
    limit: int = 5,
) -> list[RetrievedChunk]:
    """Score fused candidates and keep the best matches.

    Callers pass the list from ``retrieve`` over the existing collection and
    keyword index. Each returned chunk keeps its text and metadata. Its score
    is the cross-encoder logit. Ties break by chunk id.

    Args:
        query: User text.
        chunks: Fused candidates, in retrieval order.
        reranker: Model that scores the query against each chunk's text.
        limit: Maximum chunks to return. Defaults to 5.

    Returns:
        Reranked chunks, best first. Empty when ``chunks`` is empty or
        ``limit`` is less than 1.
    """
    if limit < 1 or not chunks:
        return []
    scores = reranker.score(query, [chunk.text for chunk in chunks])
    ranked = [
        replace(chunk, score=float(score)) for chunk, score in zip(chunks, scores, strict=True)
    ]
    ranked.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return ranked[:limit]
