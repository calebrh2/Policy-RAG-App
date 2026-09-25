"""Generation cites chunk sources and refuses when no citation survives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from rag.generate import REFUSAL, generate
from rag.retrieve import RetrievedChunk


class FakeChat:
    """Returns a fixed assistant text and records the messages it received."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.messages.append(messages)
        return self.raw


def _chunk(chunk_id: str, text: str, *, title: str = "Plastic Policy", section: str = "Rules") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"document_title": title, "section_path": section},
        score=0.0,
    )


def _answer(supported: bool, text: str, citations: list[dict[str, str]]) -> str:
    return json.dumps({"supported": supported, "text": text, "citations": citations})


def test_empty_chunks_refuse_without_calling_the_model() -> None:
    model = FakeChat(_answer(True, "unused", []))

    answer = generate("travel expenses", [], model)

    assert answer.supported is False
    assert answer.text == REFUSAL
    assert answer.citations == []
    assert model.messages == []


def test_known_chunk_keeps_document_title_and_section() -> None:
    chunk = _chunk(
        "cups",
        "Plastic Policy\nRules\nSingle-use plastic cups are banned.",
        title="Plastic Policy",
        section="Rules",
    )
    model = FakeChat(
        _answer(True, "Single-use plastic cups are banned.", [{"chunk_id": "cups"}])
    )

    answer = generate("Are plastic cups banned?", [chunk], model)

    assert answer.supported is True
    assert answer.text == "Single-use plastic cups are banned."
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.chunk_id == "cups"
    assert citation.document_title == "Plastic Policy"
    assert citation.section_path == "Rules"
    assert "quote" not in citation.model_dump()


def test_unknown_chunk_id_is_dropped() -> None:
    chunk = _chunk("cups", "plastic cups are banned")
    model = FakeChat(_answer(True, "Banned.", [{"chunk_id": "invented"}]))

    answer = generate("cups", [chunk], model)

    assert answer.supported is False
    assert answer.text == REFUSAL
    assert answer.citations == []


def test_missing_document_title_is_dropped() -> None:
    chunk = _chunk("cups", "plastic cups are banned", title="")
    model = FakeChat(_answer(True, "Banned.", [{"chunk_id": "cups"}]))

    answer = generate("cups", [chunk], model)

    assert answer.supported is False
    assert answer.text == REFUSAL
    assert answer.citations == []


def test_supported_false_returns_the_refusal() -> None:
    chunk = _chunk("water", "Sites record monthly water use.")
    model = FakeChat(
        _answer(
            False,
            "Maybe next week.",
            [{"chunk_id": "water"}],
        )
    )

    answer = generate("travel expenses", [chunk], model)

    assert answer.supported is False
    assert answer.text == REFUSAL
    assert answer.citations == []


def test_string_citations_are_accepted() -> None:
    chunk = _chunk("cups", "plastic cups are banned")
    model = FakeChat(
        json.dumps(
            {
                "supported": True,
                "text": "Plastic cups are banned.",
                "citations": ["cups"],
            }
        )
    )

    answer = generate("cups", [chunk], model)

    assert answer.supported is True
    assert answer.text == "Plastic cups are banned."
    assert answer.citations[0].chunk_id == "cups"
    assert answer.citations[0].document_title == "Plastic Policy"


def test_invalid_json_raises() -> None:
    chunk = _chunk("cups", "plastic cups are banned")
    model = FakeChat("not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        generate("cups", [chunk], model)


def test_prompt_sandwiches_chunks_and_repeats_the_question() -> None:
    chunks = [_chunk(chunk_id, chunk_id) for chunk_id in ("rank1", "rank2", "rank3", "rank4")]
    model = FakeChat(_answer(False, REFUSAL, []))

    generate("where do cups go", chunks, model)

    prompt = model.messages[0][1]["content"].split("</examples>", maxsplit=1)[1]
    expected = ["rank1", "rank3", "rank4", "rank2"]
    positions = [prompt.index(f"chunk_id: {chunk_id}") for chunk_id in expected]
    assert positions == sorted(positions)
    first_question = prompt.index("<question>")
    chunks_at = prompt.index("<chunks>")
    last_question = prompt.rindex("<question>")
    assert first_question < chunks_at < last_question
    assert prompt.count("<question>\nwhere do cups go\n</question>") == 2
    assert "quote" not in model.messages[0][1]["content"]
