"""Choose whether a question needs the current plan, the outdated plan, or both."""

from __future__ import annotations

import re

from rag.adapters.ollama import LanguageModel
from rag.retrieval import RetrievalHit

_LABELS = {"current", "outdated", "compare"}
_ROUTE_PROMPT = (
    "Choose one route for the question.\n"
    "Reply with one word only: current, outdated, or compare.\n"
    "current: a fact from the latest plan.\n"
    "outdated: a fact from an older plan.\n"
    "compare: what changed between the latest plan and an older plan.\n"
    "Question: {question}"
)

_COMPARE = (
    "difference",
    "different",
    "changed",
    "change",
    "compared",
    "compare",
    "versus",
    "vs",
    "update",
    "updated",
    "removed",
    "added",
    "both versions",
    "old and new",
    "from before",
)
_OUTDATED = (
    "outdated",
    "old version",
    "previous version",
    "test version",
    "previously",
    "old plan",
    "old carbon",
)


def route(question: str, model: LanguageModel | None = None) -> str:
    """Use the word list first. Ask the model only when those words do not match."""
    text = question.casefold()
    if _matches(text, _COMPARE):
        return "compare"
    if _matches(text, _OUTDATED):
        return "outdated"
    if model is None:
        return "current"
    label = model.complete(_ROUTE_PROMPT.format(question=question)).strip().casefold()
    if label in _LABELS:
        return label
    return "current"


def comparison_hits(
    current: list[RetrievalHit],
    outdated: list[RetrievalHit],
    *,
    limit: int = 5,
) -> list[RetrievalHit]:
    """Pair chunks that share a section name. One side missing means added or removed."""
    if not outdated:
        return current[:limit]
    by_current = {hit.section: hit for hit in current}
    by_outdated = {hit.section: hit for hit in outdated}
    both = [section for section in by_current if section in by_outdated]
    added = [section for section in by_current if section not in by_outdated]
    removed = [section for section in by_outdated if section not in by_current]
    paired = [
        _pair(by_current.get(section), by_outdated.get(section))
        for section in (both + added + removed)[:limit]
    ]
    return paired


def _matches(text: str, phrases: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)


def _pair(current: RetrievalHit | None, outdated: RetrievalHit | None) -> RetrievalHit:
    chosen = current or outdated
    assert chosen is not None
    return RetrievalHit(
        chunk_id=chosen.chunk_id,
        document_name=chosen.document_name,
        version="compare",
        section=chosen.section,
        source_pages=_pages(current, outdated),
        text=_body(current, outdated),
        score=0.0,
    )


def _pages(current: RetrievalHit | None, outdated: RetrievalHit | None) -> str:
    parts: list[str] = []
    if current is not None:
        parts.append(f"current {current.source_pages}")
    if outdated is not None:
        parts.append(f"outdated {outdated.source_pages}")
    return "; ".join(parts)


def _body(current: RetrievalHit | None, outdated: RetrievalHit | None) -> str:
    current_text = current.text if current is not None else "This section is not in the current plan."
    outdated_text = outdated.text if outdated is not None else "This section is not in the outdated plan."
    return f"Current:\n{current_text}\n\nOutdated:\n{outdated_text}"
