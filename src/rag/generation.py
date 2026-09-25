"""Turn reranked chunks into an answer with citations."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.adapters.ollama import LanguageModel
from rag.models import Answer, Citation
from rag.retrieval import RetrievalHit

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%")


class GenerationError(Exception):
    """The model reply is not an answer grounded in the retrieved chunks."""


class _ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[int]


def generate(question: str, hits: list[RetrievalHit], model: LanguageModel) -> Answer:
    """Ask the local model to cite chunk numbers, then fill in the section names."""
    if not hits:
        return Answer(text="The policy documents do not contain this.", citations=[])
    raw = model.complete(_prompt(question, hits))
    reply = _parse(raw)
    answer = Answer(text=reply.text, citations=_citations(reply.citations, hits))
    _check_numbers(answer, hits)
    return answer


def _prompt(question: str, hits: list[RetrievalHit]) -> str:
    chunks = "\n\n".join(_chunk(index, hit) for index, hit in enumerate(hits, start=1))
    return (
        "Answer the question using only the chunks below.\n"
        'Reply with JSON only: {"text": "...", "citations": [1]}\n'
        "citations is the list of chunk numbers, such as 1 for the chunk labeled [1].\n"
        "Copy numbers exactly as written in the chunks. Do not calculate.\n"
        "When a chunk shows Current and Outdated, report only facts that differ. "
        "If both sides say the same thing, do not call it a change.\n"
        'If the chunks do not contain the answer, use text '
        '"The policy documents do not contain this." and citations [].\n\n'
        f"Question: {question}\n\n"
        f"Chunks:\n{chunks}"
    )


def _chunk(index: int, hit: RetrievalHit) -> str:
    return (
        f"[{index}] document_name: {hit.document_name}\n"
        f"section: {hit.section}\n"
        f"source_pages: {hit.source_pages}\n"
        f"{hit.text}"
    )


def _parse(raw: str) -> _ModelReply:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.removesuffix("```").strip()
    try:
        return _ModelReply.model_validate_json(text)
    except ValidationError as exc:
        message = "The model did not return the answer JSON."
        raise GenerationError(message) from exc


def _citations(indexes: list[int], hits: list[RetrievalHit]) -> list[Citation]:
    citations: list[Citation] = []
    for index in indexes:
        if index < 1 or index > len(hits):
            message = f"Citation {index} is not one of the retrieved chunks."
            raise GenerationError(message)
        hit = hits[index - 1]
        citations.append(
            Citation(
                document_name=hit.document_name,
                section=hit.section,
                source_pages=hit.source_pages,
            )
        )
    return citations


def _check_numbers(answer: Answer, hits: list[RetrievalHit]) -> None:
    cited = [
        hit
        for hit in hits
        if (hit.document_name, hit.section, hit.source_pages)
        in {(item.document_name, item.section, item.source_pages) for item in answer.citations}
    ]
    allowed = {number for hit in cited for number in _NUMBER.findall(hit.text)}
    for number in _NUMBER.findall(answer.text):
        if number not in allowed:
            message = f"{number} is not in the cited chunks."
            raise GenerationError(message)
