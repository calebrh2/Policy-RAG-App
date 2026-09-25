"""Answer a query from reranked chunks, with checked citations.

Purpose
-------
Builds a delimited few-shot prompt, calls a chat model, and keeps a citation
when the model names a supplied chunk that has a document title. Document
title and section come from the chunk's metadata. An answer with no surviving
citation is a refusal.

Contents
--------
- ``Citation``: one source that informed the answer.
- ``Answer``: the validated reply.
- ``generate``: prompt, parse, and check citations.

The system message, instructions, and few-shot examples are loaded from
``prompts/generation-prompt-v1.md``. ``_PROMPT_FILE`` selects the file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.adapters.chat import ChatModel
from rag.retrieve import RetrievedChunk

REFUSAL = "The corpus does not mention this."

# Swap by adding prompts/generation-prompt-v2.md and changing this name.
_PROMPT_FILE = "generation-prompt-v1"

_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)


class Citation(BaseModel):
    """One chunk that informed the answer."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_title: str
    section_path: str


class Answer(BaseModel):
    """A validated reply. Citations are empty when the corpus does not answer."""

    model_config = ConfigDict(frozen=True)

    supported: bool
    text: str
    citations: list[Citation]


class _DraftCitation(BaseModel):
    """A citation as returned by the model, before source checks."""

    chunk_id: str


class _DraftAnswer(BaseModel):
    """The JSON object the model returns."""

    supported: bool
    text: str
    citations: list[_DraftCitation]


def _load_prompt(name: str) -> tuple[str, str, str]:
    """Load the system message, instructions, and examples for one prompt file.

    Args:
        name: File stem under ``prompts/``, such as ``generation-prompt-v1``.

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


_SYSTEM, _INSTRUCTIONS, _EXAMPLES = _load_prompt(_PROMPT_FILE)


def generate(query: str, chunks: Sequence[RetrievedChunk], model: ChatModel) -> Answer:
    """Answer one query from reranked chunks.

    An empty chunk list returns the refusal without calling the model. The
    prompt places the question before and after the chunks. Inside the chunk
    section, rerank order is rearranged so the best chunk is first and the
    second best is last. Each surviving citation copies ``document_title`` and
    ``section_path`` from the chunk.

    Args:
        query: User text.
        chunks: Reranked chunks, best first.
        model: Chat model that returns a JSON answer.

    Returns:
        A supported answer, or the refusal when no citation survives.

    Raises:
        ValueError: The model returned invalid JSON or the wrong shape.
    """
    if not chunks:
        return _refusal()
    draft = _parse(model.complete(_messages(query, chunks)))
    if not draft.supported:
        return _refusal()
    citations = _checked_citations(draft.citations, chunks)
    if not citations:
        return _refusal()
    return Answer(supported=True, text=draft.text, citations=citations)


def _refusal() -> Answer:
    """Return the fixed refusal.

    Returns:
        An unsupported answer with no citations.
    """
    return Answer(supported=False, text=REFUSAL, citations=[])


def _messages(query: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the system and user messages.

    Args:
        query: User text.
        chunks: Reranked chunks, best first.

    Returns:
        One system message and one delimited user message.
    """
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(query, chunks)},
    ]


def _user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build the delimited user message.

    Args:
        query: User text, placed before and after the chunks.
        chunks: Reranked chunks, best first.

    Returns:
        Instructions, examples, the question, sandwiched chunks, and the
        question again.
    """
    question = f"<question>\n{query}\n</question>"
    return "\n\n".join(
        [
            f"<instructions>\n{_INSTRUCTIONS}\n</instructions>",
            f"<examples>\n{_EXAMPLES}\n</examples>",
            question,
            f"<chunks>\n{_render_chunks(_sandwich(chunks))}\n</chunks>",
            question,
        ]
    )


def _sandwich(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Place the best chunk first and the second best last.

    Even positions in rerank order fill from the front. Odd positions fill
    from the back. The third best therefore sits second, and the fourth sits
    second to last.

    Args:
        chunks: Reranked chunks, best first.

    Returns:
        The same chunks in sandwich order.
    """
    placed: list[RetrievedChunk | None] = [None] * len(chunks)
    front = 0
    back = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            placed[front] = chunk
            front += 1
        else:
            placed[back] = chunk
            back -= 1
    return [chunk for chunk in placed if chunk is not None]


def _render_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks with their ids and source metadata.

    Args:
        chunks: Chunks in prompt order.

    Returns:
        One ``<chunk>`` block per chunk.
    """
    blocks = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    "<chunk>",
                    f"chunk_id: {chunk.chunk_id}",
                    f"document_title: {_meta(chunk.metadata, 'document_title')}",
                    f"section_path: {_meta(chunk.metadata, 'section_path')}",
                    chunk.text,
                    "</chunk>",
                ]
            )
        )
    return "\n".join(blocks)


def _meta(metadata: Mapping[str, str | bool], key: str) -> str:
    """Return one string metadata field.

    Args:
        metadata: Chunk metadata.
        key: Field name.

    Returns:
        The string value, or an empty string when the field is missing or
        not a string.
    """
    value = metadata.get(key, "")
    return value if isinstance(value, str) else ""


def _parse(raw: str) -> _DraftAnswer:
    """Parse the model text as an answer.

    Args:
        raw: Assistant message text.

    Returns:
        The draft answer.

    Raises:
        ValueError: The text is not JSON, or it does not match the schema.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Model returned invalid JSON.") from exc
    try:
        return _DraftAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Model returned JSON that does not match the answer schema.") from exc


def _checked_citations(
    drafts: Sequence[_DraftCitation], chunks: Sequence[RetrievedChunk]
) -> list[Citation]:
    """Keep citations that name a supplied chunk with a document title.

    Args:
        drafts: Citations from the model.
        chunks: Chunks that were sent in the prompt.

    Returns:
        Citations with document title and section copied from chunk metadata.
        A citation is omitted when the id is unknown or the title is missing.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citations: list[Citation] = []
    for draft in drafts:
        chunk = by_id.get(draft.chunk_id)
        if chunk is None:
            continue
        title = _meta(chunk.metadata, "document_title")
        if not title:
            continue
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_title=title,
                section_path=_meta(chunk.metadata, "section_path"),
            )
        )
    return citations
