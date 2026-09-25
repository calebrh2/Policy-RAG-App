"""API for the policy chat UI.

Purpose
-------
Serves a health check, the existing ingest echo, and ``POST /query``. The
query route runs hybrid retrieval, cross-encoder reranking, and generation.
The response text and citations are what the chat shows.

Contents
--------
- ``app``: the FastAPI application.
- ``answer_query``: retrieve, rerank, and generate for one question.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.adapters.chat import ChatModel
from rag.adapters.reranker import Reranker
from rag.generate import Answer
from rag.retrieve import ChunkSearch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_collection: ChunkSearch | None = None
_reranker: Reranker | None = None
_model: ChatModel | None = None


class IngestRequest(BaseModel):
    """Text posted to the ingest echo."""

    text: str


class QueryRequest(BaseModel):
    """One user question for the policy corpus."""

    text: str


@app.get("/")
def root() -> dict[str, str]:
    """Report that the API is up.

    Returns:
        A status object.
    """
    return {"status": "ok"}


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict[str, bool | str]:
    """Echo posted text.

    Args:
        body: Text from the client.

    Returns:
        Whether the body was received, and the text.
    """
    return {"ok": True, "received": body.text}


@app.post("/query")
def query(body: QueryRequest) -> Answer:
    """Answer one question from the policy corpus.

    Args:
        body: The user's question.

    Returns:
        The generated answer and the sources that informed it.

    Raises:
        HTTPException: The question is blank, or the model endpoint failed.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Query text is empty.")
    try:
        return answer_query(text)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def answer_query(query_text: str) -> Answer:
    """Retrieve, rerank, and generate an answer.

    The dense collection, reranker, and chat model are loaded on first use
    and reused. The keyword index is opened for this call.

    Args:
        query_text: User question.

    Returns:
        The validated answer, including document title and section on each
        citation.
    """
    from rag.adapters.chat import default_chat_model
    from rag.ChromaDB import get_collection
    from rag.generate import generate
    from rag.keyword_index import KeywordIndex
    from rag.rerank import default_reranker, rerank
    from rag.retrieve import retrieve

    global _collection, _reranker, _model
    with _lock:
        if _collection is None:
            _collection = get_collection()
        if _reranker is None:
            _reranker = default_reranker()
        if _model is None:
            _model = default_chat_model()
        with KeywordIndex() as index:
            chunks = retrieve(query_text, _collection, index)
        return generate(query_text, rerank(query_text, chunks, _reranker), _model)
