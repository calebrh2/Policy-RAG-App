"""Golden questions for each named corpus.

Purpose
-------
``cases_for`` returns the answerable questions, refusals, and generation set
for ``meridian`` or ``coforge``.

Contents
--------
- ``EvalSet``: the three tuples for one corpus.
- ``cases_for``: look up a corpus by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.evaluation.cases import CASES, GENERATION_CASES, REFUSAL_CASES, EvalCase
from tests.evaluation.meridian_cases import (
    MERIDIAN_CASES,
    MERIDIAN_GENERATION_CASES,
    MERIDIAN_REFUSAL_CASES,
)


@dataclass(frozen=True)
class EvalSet:
    """Answerable questions, refusals, and the generation list for one corpus."""

    cases: tuple[EvalCase, ...]
    refusal_cases: tuple[EvalCase, ...]
    generation_cases: tuple[EvalCase, ...]


_SETS = {
    "coforge": EvalSet(CASES, REFUSAL_CASES, GENERATION_CASES),
    "meridian": EvalSet(MERIDIAN_CASES, MERIDIAN_REFUSAL_CASES, MERIDIAN_GENERATION_CASES),
}


def cases_for(name: str) -> EvalSet:
    """Return the golden questions for one corpus.

    Args:
        name: ``meridian`` or ``coforge``.

    Returns:
        That corpus's evaluation set.

    Raises:
        ValueError: ``name`` is not a known corpus.
    """
    found = _SETS.get(name)
    if found is None:
        known = ", ".join(_SETS)
        raise ValueError(f"Unknown corpus {name!r}. Known corpora: {known}")
    return found
