"""Generated answers must contain the expected key facts.

Purpose
-------
The fixed set names the phrases each answer must include. These tests check
that rule on the gold answers and on answers that omit a fact. The generation
script applies the same check to model replies and records it.

Contents
--------
- ``test_set_has_at_least_eight_answerable_questions``: the fixed set is large
  enough and each gold answer contains its key facts.
- ``test_missing_key_fact_fails``: an answer that drops a required phrase fails.
- ``test_key_fact_match_is_case_insensitive``: capitalization does not matter.
- ``test_refusal_must_match_the_refusal_exactly``: an unanswerable question
  passes only on the fixed refusal.
"""

from __future__ import annotations

from rag.generate import REFUSAL
from rag.judge import contains_key_information
from tests.evaluation.cases import CASES, REFUSAL_CASES


def test_set_has_at_least_eight_answerable_questions() -> None:
    """Each answerable question has key facts, and the gold answer contains them."""
    assert len(CASES) >= 8
    for case in CASES:
        assert case.key_facts
        assert contains_key_information(case.gold_answer, case.key_facts)


def test_missing_key_fact_fails() -> None:
    """An answer that omits a required phrase does not pass."""
    case = next(case for case in CASES if case.case_id == "carbon-reduction-target")

    assert not contains_key_information("Emissions will fall by 2030.", case.key_facts)


def test_key_fact_match_is_case_insensitive() -> None:
    """A change in capitalization still contains the key fact."""
    case = next(case for case in CASES if case.case_id == "plastic-cups-prohibited")

    assert contains_key_information(
        "Yes. PLASTIC PLATES, CUPS, AND GLASSES are prohibited.",
        case.key_facts,
    )


def test_refusal_must_match_the_refusal_exactly() -> None:
    """An unanswerable question passes only when the text is the refusal."""
    for case in REFUSAL_CASES:
        assert contains_key_information(REFUSAL, case.key_facts, exact=True)
        assert not contains_key_information(
            REFUSAL + " The deadline is Friday.",
            case.key_facts,
            exact=True,
        )
