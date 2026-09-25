"""Faithfulness and accuracy judgments, plus citation and refusal checks.

Purpose
-------
Scores a generated answer. Citation and refusal checks are deterministic.
Faithfulness and accuracy are pass/fail judgments from a chat model. The
rubrics live in versioned prompt files. ``_FAITHFULNESS_PROMPT`` and
``_ACCURACY_PROMPT`` select those files.

Contents
--------
- ``Verdict``: one judge result.
- ``citations_ok``: every cited id names a sent chunk with a document title.
- ``refusal_ok``: an unsupported answer is the fixed refusal.
- ``raw_citation_ids``: chunk ids from a generation completion.
- ``judge_faithfulness``: judge the answer against the chunks it cites.
- ``judge_accuracy``: judge the answer against the gold answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.adapters.chat import ChatModel
from rag.generate import REFUSAL, Answer
from rag.retrieve import RetrievedChunk

# Swap by adding prompts/faithfulness-prompt-v2.md and changing this name.
_FAITHFULNESS_PROMPT = "faithfulness-prompt-v1"
# Swap by adding prompts/accuracy-prompt-v2.md and changing this name.
_ACCURACY_PROMPT = "accuracy-prompt-v1"

_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
_INVALID_JSON = "Judge returned invalid JSON."
_INVALID_SHAPE = "Judge returned JSON that does not match the verdict schema."


@dataclass(frozen=True)
class Verdict:
    """One judge result. ``reason`` is empty when ``passed`` is true."""

    passed: bool
    reason: str


class _ParsedVerdict(BaseModel):
    """The JSON object the judge returns."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    passed: bool = Field(alias="pass")
    reason: str = ""


class _RawCitation(BaseModel):
    """One citation object in a generation completion."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str


class _RawAnswer(BaseModel):
    """The citation list from a generation completion."""

    model_config = ConfigDict(frozen=True)

    citations: list[_RawCitation]


def citations_ok(citation_ids: Sequence[str], chunks: Sequence[RetrievedChunk]) -> bool:
    """Return whether every cited id names a sent chunk with a document title.

    An empty id list passes. A missing id or a blank title fails the list.

    Args:
        citation_ids: Chunk ids from the model, before dropping invalid ones.
        chunks: Chunks that were sent in the generation prompt.

    Returns:
        True when every id is in ``chunks`` and that chunk has a non-empty
        ``document_title``.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for citation_id in citation_ids:
        chunk = by_id.get(citation_id)
        if chunk is None:
            return False
        title = chunk.metadata.get("document_title", "")
        if not isinstance(title, str) or not title:
            return False
    return True


def refusal_ok(answer: Answer) -> bool:
    """Return whether an unsupported answer has the fixed refusal shape.

    A supported answer passes. An unsupported answer passes only when its text
    is the refusal and it has no citations.

    Args:
        answer: The validated reply from ``generate``.

    Returns:
        True when the answer is supported, or when it is the fixed refusal.
    """
    if answer.supported:
        return True
    return answer.text == REFUSAL and not answer.citations


def raw_citation_ids(raw: str) -> list[str]:
    """Return chunk ids from a generation completion.

    Args:
        raw: Assistant message text from the generation call.

    Returns:
        Citation ids in the order the model emitted them.

    Raises:
        ValueError: The text is not JSON, or it has no citations list.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Generation completion is not JSON.") from exc
    try:
        answer = _RawAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Generation completion has no citations list.") from exc
    return [citation.chunk_id for citation in answer.citations]


def judge_faithfulness(
    question: str,
    answer: Answer,
    chunks: Sequence[RetrievedChunk],
    model: ChatModel,
) -> Verdict:
    """Judge whether the answer is supported by the chunks it cites.

    Chunks the answer does not cite are left out of the prompt.

    Args:
        question: User text.
        answer: The validated reply from ``generate``.
        chunks: Chunks that were sent in the generation prompt.
        model: Chat model that returns a JSON verdict.

    Returns:
        A pass/fail verdict. A model or JSON failure is a failed verdict.
    """
    system, instructions, examples = _FAITHFULNESS
    body = _faithfulness_body(question, answer.text, _cited(answer, chunks))
    return _judge(model, _messages(system, instructions, examples, body))


def judge_accuracy(
    question: str,
    answer: Answer,
    gold_answer: str,
    model: ChatModel,
) -> Verdict:
    """Judge whether the answer matches the gold answer.

    The prompt contains the question, the reference, and the answer. It does
    not contain source chunks.

    Args:
        question: User text.
        answer: The validated reply from ``generate``.
        gold_answer: The reference reply for this question.
        model: Chat model that returns a JSON verdict.

    Returns:
        A pass/fail verdict. A model or JSON failure is a failed verdict.
    """
    system, instructions, examples = _ACCURACY
    body = "\n\n".join(
        [
            f"<question>\n{question}\n</question>",
            f"<reference>\n{gold_answer}\n</reference>",
            f"<answer>\n{answer.text}\n</answer>",
        ]
    )
    return _judge(model, _messages(system, instructions, examples, body))


def _cited(answer: Answer, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return the sent chunks named by the answer, in citation order.

    Args:
        answer: The validated reply.
        chunks: Chunks that were sent in the generation prompt.

    Returns:
        Cited chunks. An id that was not sent is omitted.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [by_id[citation.chunk_id] for citation in answer.citations if citation.chunk_id in by_id]


def _faithfulness_body(question: str, answer_text: str, cited: Sequence[RetrievedChunk]) -> str:
    """Build the faithfulness case: question, cited chunks, and answer.

    Args:
        question: User text.
        answer_text: The reply being judged.
        cited: Chunks named by the answer.

    Returns:
        The delimited case body.
    """
    return "\n\n".join(
        [
            f"<question>\n{question}\n</question>",
            f"<cited_chunks>\n{_render_cited(cited)}\n</cited_chunks>",
            f"<answer>\n{answer_text}\n</answer>",
        ]
    )


def _render_cited(chunks: Sequence[RetrievedChunk]) -> str:
    """Render cited chunks with their ids.

    Args:
        chunks: Cited chunks, in citation order.

    Returns:
        One ``<chunk>`` block per chunk, or an empty string when there are none.
    """
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append("\n".join(["<chunk>", f"chunk_id: {chunk.chunk_id}", chunk.text, "</chunk>"]))
    return "\n".join(blocks)


def _messages(system: str, instructions: str, examples: str, body: str) -> list[dict[str, str]]:
    """Build the system and user messages for one judgment.

    Args:
        system: System message text.
        instructions: Rubric text.
        examples: Few-shot verdicts.
        body: The case being judged.

    Returns:
        One system message and one delimited user message.
    """
    user = "\n\n".join(
        [
            f"<instructions>\n{instructions}\n</instructions>",
            f"<examples>\n{examples}\n</examples>",
            body,
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _judge(model: ChatModel, messages: Sequence[Mapping[str, str]]) -> Verdict:
    """Call the judge and parse its verdict.

    Args:
        model: Chat model that returns a JSON verdict.
        messages: System and user messages.

    Returns:
        The parsed verdict. A model error or invalid JSON is a failure.
    """
    try:
        raw = model.complete(messages)
    except ValueError as exc:
        return Verdict(passed=False, reason=str(exc))
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> Verdict:
    """Parse a judge completion.

    Args:
        raw: Assistant message text.

    Returns:
        The verdict. Invalid JSON or the wrong shape is a failed verdict.
        A passing verdict stores an empty reason.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return Verdict(passed=False, reason=_INVALID_JSON)
    try:
        parsed = _ParsedVerdict.model_validate(payload)
    except ValidationError:
        return Verdict(passed=False, reason=_INVALID_SHAPE)
    if parsed.passed:
        return Verdict(passed=True, reason="")
    return Verdict(passed=False, reason=parsed.reason)


def _load_prompt(name: str) -> tuple[str, str, str]:
    """Load the system message, instructions, and examples for one prompt file.

    Args:
        name: File stem under ``prompts/``, such as ``faithfulness-prompt-v1``.

    Returns:
        System text, instruction text, and example text.

    Raises:
        ValueError: The file is missing a System, Instructions, or Examples
        section.
    """
    path = Path(__file__).parent / "prompts" / f"{name}.md"
    sections = _prompt_sections(path.read_text(encoding="utf-8"))
    try:
        return sections["System"], sections["Instructions"], sections["Examples"]
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Prompt file {path.name} is missing a {missing} section.") from exc


def _prompt_sections(text: str) -> dict[str, str]:
    """Split a prompt file on ``## `` headings.

    Args:
        text: Markdown prompt file.

    Returns:
        Heading text mapped to the stripped body that follows it.
    """
    matches = list(_HEADING.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()
    return sections


_FAITHFULNESS = _load_prompt(_FAITHFULNESS_PROMPT)
_ACCURACY = _load_prompt(_ACCURACY_PROMPT)
