"""Citation checks, refusal shape, and judge prompt parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from rag.generate import REFUSAL, Answer, Citation
from rag.judge import (
    _ACCURACY_PROMPT,
    _FAITHFULNESS_PROMPT,
    citations_ok,
    judge_accuracy,
    judge_faithfulness,
    raw_citation_ids,
    refusal_ok,
)
from rag.retrieve import RetrievedChunk
from tests.evaluation.catalog import cases_for


class FakeChat:
    """Returns a fixed assistant text and records the messages it received."""

    def __init__(self, raw: str) -> None:
        """Store the text ``complete`` will return.

        Args:
            raw: Assistant message text.
        """
        self.raw = raw
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Record the messages and return the fixed text.

        Args:
            messages: Role and content pairs, in conversation order.

        Returns:
            The fixed assistant text.
        """
        self.messages.append(messages)
        return self.raw


def _chunk(chunk_id: str, text: str, *, title: str = "Plastic Policy") -> RetrievedChunk:
    """Return one retrieved chunk.

    Args:
        chunk_id: Chunk id.
        text: Chunk body.
        title: Document title stored in metadata.

    Returns:
        A chunk with a zero score.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"document_title": title, "section_path": "Rules"},
        score=0.0,
    )


def _answer(text: str, chunk_ids: list[str], *, supported: bool = True) -> Answer:
    """Return an answer that cites the given ids.

    Args:
        text: Reply text.
        chunk_ids: Cited chunk ids.
        supported: Whether the answer claims the corpus supports it.

    Returns:
        A validated answer.
    """
    return Answer(
        supported=supported,
        text=text,
        citations=[
            Citation(chunk_id=chunk_id, document_title="Plastic Policy", section_path="Rules")
            for chunk_id in chunk_ids
        ],
    )


def _user_text(model: FakeChat) -> str:
    """Return the user message from the single judge call.

    Args:
        model: Fake chat that was asked to judge once.

    Returns:
        The user message content.
    """
    return model.messages[0][1]["content"]


def test_prompt_files_are_version_one() -> None:
    """The judge selects the v1 faithfulness and accuracy prompts."""
    assert _FAITHFULNESS_PROMPT == "faithfulness-prompt-v1"
    assert _ACCURACY_PROMPT == "accuracy-prompt-v1"


def test_answerable_cases_have_gold_answers() -> None:
    """Each retrieval case expects a supported answer with a reference."""
    for name in ("meridian", "coforge"):
        cases = cases_for(name).cases
        assert cases
        for case in cases:
            assert case.expect_supported
            assert case.gold_answer
            assert case.gold_answer != REFUSAL


def test_refusal_cases_are_unanswerable() -> None:
    """Each generation set adds questions that corpus cannot answer."""
    for name in ("meridian", "coforge"):
        evaluation = cases_for(name)
        assert len(evaluation.refusal_cases) == 2
        for case in evaluation.refusal_cases:
            assert not case.expect_supported
            assert case.gold_answer == REFUSAL
            assert case.document_id == ""
            assert case not in evaluation.cases
            assert case in evaluation.generation_cases


def test_known_citation_with_a_title_passes() -> None:
    """A sent chunk id with a document title is a valid citation."""
    chunks = [_chunk("cups", "Plastic cups are prohibited.")]

    assert citations_ok(["cups"], chunks)


def test_empty_citation_list_passes() -> None:
    """No cited ids is a valid citation list."""
    assert citations_ok([], [_chunk("cups", "Plastic cups are prohibited.")])


def test_unknown_citation_id_fails() -> None:
    """An id that was not sent fails the citation check."""
    chunks = [_chunk("cups", "Plastic cups are prohibited.")]

    assert not citations_ok(["invented"], chunks)


def test_missing_document_title_fails() -> None:
    """A sent chunk with a blank title fails the citation check."""
    chunks = [_chunk("cups", "Plastic cups are prohibited.", title="")]

    assert not citations_ok(["cups"], chunks)


def test_raw_citation_ids_preserve_order() -> None:
    """Citation ids are read from the generation completion in order."""
    raw = json.dumps(
        {
            "supported": True,
            "text": "Yes.",
            "citations": [{"chunk_id": "cups"}, {"chunk_id": "plates"}],
        }
    )

    assert raw_citation_ids(raw) == ["cups", "plates"]


def test_raw_citation_ids_accept_a_string_id() -> None:
    """A citation that is only the chunk id is kept."""
    raw = json.dumps({"supported": True, "text": "Yes.", "citations": ["cups"]})

    assert raw_citation_ids(raw) == ["cups"]


def test_raw_citation_ids_reject_invalid_json() -> None:
    """A completion that is not an answer JSON raises."""
    with pytest.raises(ValueError, match="not JSON"):
        raw_citation_ids("not json")


def test_supported_answer_passes_refusal_shape() -> None:
    """A supported answer is not required to be the refusal."""
    assert refusal_ok(_answer("Yes.", ["cups"]))


def test_fixed_refusal_passes_refusal_shape() -> None:
    """The fixed refusal text with no citations passes."""
    assert refusal_ok(_answer(REFUSAL, [], supported=False))


def test_other_unsupported_text_fails_refusal_shape() -> None:
    """An unsupported answer that is not the refusal fails."""
    assert not refusal_ok(_answer("Maybe next week.", [], supported=False))


def test_unsupported_answer_with_a_citation_fails_refusal_shape() -> None:
    """An unsupported answer that still cites a chunk fails."""
    assert not refusal_ok(_answer(REFUSAL, ["cups"], supported=False))


def test_faithfulness_prompt_contains_only_cited_chunk_text() -> None:
    """The faithfulness judge sees the cited chunk and not an uncited one."""
    cited = _chunk("cited-id-xyz", "CITED-FACT-ALPHA")
    uncited = _chunk("uncited-id-xyz", "UNCITED-FACT-BETA")
    model = FakeChat('{"pass": true}')

    verdict = judge_faithfulness(
        "Are cups banned?",
        _answer("Yes.", ["cited-id-xyz"]),
        [cited, uncited],
        model,
    )

    prompt = _user_text(model)
    assert "CITED-FACT-ALPHA" in prompt
    assert "cited-id-xyz" in prompt
    assert "UNCITED-FACT-BETA" not in prompt
    assert "uncited-id-xyz" not in prompt
    assert verdict.passed
    assert verdict.reason == ""


def test_accuracy_prompt_contains_the_gold_answer_and_no_chunks() -> None:
    """The accuracy judge sees the reference and no source chunks."""
    model = FakeChat('{"pass": false, "reason": "missing the reference fact"}')

    verdict = judge_accuracy(
        "By when?",
        _answer("By Dec 2025.", ["cups"]),
        "REFERENCE-FACT-GAMMA",
        model,
    )

    prompt = _user_text(model)
    assert "REFERENCE-FACT-GAMMA" in prompt
    assert "<cited_chunks>" not in prompt
    assert "<chunks>" not in prompt
    assert not verdict.passed
    assert verdict.reason == "missing the reference fact"


def test_invalid_judge_json_fails_the_verdict() -> None:
    """Invalid judge JSON is a failed verdict and does not raise."""
    model = FakeChat("not json")

    verdict = judge_accuracy("By when?", _answer(REFUSAL, [], supported=False), REFUSAL, model)

    assert not verdict.passed
    assert verdict.reason == "Judge returned invalid JSON."


def test_judge_schema_mismatch_fails_the_verdict() -> None:
    """A JSON object without pass is a failed verdict."""
    model = FakeChat('{"reason": "missing pass"}')

    verdict = judge_faithfulness("Are cups banned?", _answer("Yes.", []), [], model)

    assert not verdict.passed
    assert verdict.reason == "Judge returned JSON that does not match the verdict schema."


class _FailingChat:
    """Raises the chat adapter's request error."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Raise a request failure.

        Args:
            messages: Role and content pairs, in conversation order.

        Raises:
            ValueError: The chat request failed.
        """
        raise ValueError("Chat completion request failed: refused")


def test_judge_request_failure_fails_the_verdict() -> None:
    """A chat request error is a failed verdict and does not raise."""
    verdict = judge_accuracy("By when?", _answer(REFUSAL, [], supported=False), REFUSAL, _FailingChat())

    assert not verdict.passed
    assert verdict.reason == "Chat completion request failed: refused"
