"""Split one message into retrieval steps and check that each step is covered."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.adapters.ollama import LanguageModel
from rag.retrieval import RetrievalHit

_MAX_STEPS = 2
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "must",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "with",
    }
)
_PLAN_PROMPT = (
    "Split the message into retrieval steps.\n"
    'Reply with JSON only: {{"steps": ["..."]}}\n'
    "Use one step when the message asks one thing.\n"
    "Use two steps when it asks two things.\n"
    "Each step is a standalone question. Do not add facts.\n"
    "Message: {question}"
)


class RetrievalPlan(BaseModel):
    """The questions to retrieve, in order. At most two."""

    model_config = ConfigDict(extra="forbid")

    steps: list[str]


def decompose(question: str, model: LanguageModel) -> RetrievalPlan:
    """Ask the model to split a message. A bad reply stays one step."""
    raw = model.complete(_PLAN_PROMPT.format(question=question))
    steps = [_clean(step) for step in _steps(raw)]
    kept = [step for step in steps if step][:_MAX_STEPS]
    if not kept:
        return RetrievalPlan(steps=[question])
    return RetrievalPlan(steps=kept)


def covers(question: str, hits: list[RetrievalHit]) -> bool:
    """A step is covered when one retrieved chunk contains a content word from it."""
    words = _content_words(question)
    if not words:
        return bool(hits)
    blob = " ".join(hit.text.casefold() for hit in hits)
    found = [word for word in words if word in blob]
    longest = max(words, key=len)
    if len(words) == 1:
        return bool(found)
    return longest in found and len(found) >= 2


def focus_query(question: str) -> str:
    """Content words only, for the one deterministic retry."""
    return " ".join(_content_words(question))


def _steps(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.removesuffix("```").strip()
    try:
        plan = RetrievalPlan.model_validate_json(text)
    except ValidationError:
        return []
    return plan.steps


def _clean(step: str) -> str:
    return " ".join(step.split())


def _content_words(question: str) -> list[str]:
    words: list[str] = []
    for word in _WORD.findall(question.casefold()):
        if len(word) < 4 or word in _STOP or word in words:
            continue
        words.append(word)
    return words
