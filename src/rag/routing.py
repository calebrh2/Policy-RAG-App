"""Choose whether a question needs the current plan, the outdated plan, or both."""

from __future__ import annotations

import re

from rag.adapters.ollama import LanguageModel
from rag.retrieval import RetrievalHit

_LABELS = {"current", "outdated", "compare"}
_WORD = re.compile(r"[a-z0-9]+")
_ROUTE_PROMPT = (
    "Choose one route for the question.\n"
    "Reply with one word only: current, outdated, or compare.\n"
    "current: a fact from the latest plan.\n"
    "outdated: a fact from an older plan.\n"
    "compare: what is different between two versions or dates of the same document.\n"
    "Question: {question}"
)

_COMPARE = (
    "difference",
    "different",
    "differ",
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


_GENERIC = frozenset({"policy", "plan", "copy", "free", "management", "test", "version"})


def family_name(document_name: str) -> str:
    """Group a current file and its outdated twin under one name."""
    text = document_name.casefold().replace(" copy", "")
    text = text.replace("outdated-", "").replace("-test-version", "")
    return text.strip("- ")


def version_terms(records: list[tuple[str, str]]) -> tuple[frozenset[str], frozenset[str]]:
    """Words for families that have many versions, then words for families that have one."""
    versions: dict[str, set[str]] = {}
    for document_name, version in records:
        versions.setdefault(family_name(document_name), set()).add(version)
    multi: set[str] = set()
    single: set[str] = set()
    for name, found in versions.items():
        bucket = multi if len(found) >= 2 else single
        bucket.update(_terms(name))
    return frozenset(multi), frozenset(single - multi)


def resolve_route(
    question: str,
    model: LanguageModel | None = None,
    *,
    multi_version: frozenset[str] = frozenset(),
    single_version: frozenset[str] = frozenset(),
) -> str:
    """Route one step. Compare is kept only when that family has another version."""
    chosen = route(question, model)
    if chosen != "compare":
        return chosen
    words = set(_WORD.findall(question.casefold()))
    if _mentions(words, multi_version):
        return "compare"
    if _mentions(words, single_version):
        return "current"
    return "compare"


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


def should_compare_steps(
    labels: list[str],
    steps: list[str],
    *,
    single_version: frozenset[str],
) -> bool:
    """Two steps that ask for the current and older copy of one family are one comparison."""
    if "current" not in labels or "outdated" not in labels:
        return False
    for step in steps:
        words = set(_WORD.findall(step.casefold()))
        if _mentions(words, single_version):
            return False
    return True


def versioned_hits(hits: list[RetrievalHit], multi_version: frozenset[str]) -> list[RetrievalHit]:
    """Keep chunks from families that have more than one version."""
    if not multi_version:
        return hits
    return [
        hit
        for hit in hits
        if _mentions(set(_WORD.findall(family_name(hit.document_name))), multi_version)
    ]


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
    ordered = sorted(
        both + added + removed,
        key=lambda section: _same_percentages(by_current.get(section), by_outdated.get(section)),
    )
    paired = [
        _pair(by_current.get(section), by_outdated.get(section))
        for section in ordered[:limit]
    ]
    return paired


def _mentions(words: set[str], terms: frozenset[str]) -> bool:
    return any(word.startswith(term) or term.startswith(word) for word in words for term in terms)


def _terms(name: str) -> set[str]:
    return {word for word in _WORD.findall(name) if len(word) >= 4 and word not in _GENERIC}


def _same_percentages(current: RetrievalHit | None, outdated: RetrievalHit | None) -> bool:
    if current is None or outdated is None:
        return False
    return set(re.findall(r"\d+%", current.text)) == set(re.findall(r"\d+%", outdated.text))


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
