"""Section-first chunker for parsed policy documents.

Purpose
-------
Splits a ``ParsedDocument`` on markdown heading paths. Size is a whitespace
word count so chunking does not load an embedding model.

Contents
--------
- ``Chunk``: one stored piece of a document, before ingest assigns status.
- ``Chunker``: the chunker interface.
- ``SectionChunker``: the policy section rules.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rag.adapters.parsing import ParsedBlock, ParsedDocument

PREFERRED_MAX = 450
HARD_MAX = 500
_NUMBERED = re.compile(r"^\d+\.\s")
_BULLET = re.compile(r"^-\s")
_SENTENCE = re.compile(r"(?<=[a-z]{2}[.!?])\s+", flags=re.IGNORECASE)
_APPROVAL = {"Approved by:", "Signed on behalf of the Supplier:"}
_BOILERPLATE = {"About Coforge"}


@dataclass(frozen=True)
class Chunk:
    """One chunk. ``chunk_id`` and ``status`` are filled by ingest."""

    chunk_id: str
    text: str
    document_id: str
    document_title: str
    section_path: str
    pages: str
    version: str
    content_type: str
    searchable: bool
    status: str = ""


class Chunker(Protocol):
    """Turns a parsed document into chunks."""

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split one document.

        Args:
            document: Parser output.

        Returns:
            Chunks in reading order. ``chunk_id`` and ``status`` are empty.
        """
        ...


@dataclass(frozen=True)
class _Piece:
    """A table, list, or paragraph run that packing treats as one unit."""

    kind: str
    body: str
    pages: str
    intro: str = ""


class SectionChunker:
    """Split policy blocks on ``##`` and ``###`` paths.

    Preferred size is 100 to 450 words. A chunk may be shorter when it is a
    complete section, rule, definition, target, or table. The hard maximum is
    500 words, including the repeated title and section path. A single sentence
    longer than that stays intact.
    """

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk one parsed document.

        Args:
            document: Parser output.

        Returns:
            Chunks in reading order.
        """
        chunks: list[Chunk] = []
        for blocks in _group_by_heading(document.blocks):
            chunks.extend(_chunks_for_heading(document, blocks))
        return chunks


def _group_by_heading(blocks: tuple[ParsedBlock, ...]) -> list[list[ParsedBlock]]:
    """Group consecutive blocks that share a heading path.

    Args:
        blocks: Document blocks in order.

    Returns:
        One group per heading path run.
    """
    groups: list[list[ParsedBlock]] = []
    for block in blocks:
        if not groups or groups[-1][0].heading_path != block.heading_path:
            groups.append([block])
        else:
            groups[-1].append(block)
    return groups


def _chunks_for_heading(document: ParsedDocument, blocks: list[ParsedBlock]) -> list[Chunk]:
    """Pack one heading path into chunks.

    Args:
        document: Source document, used for the title prefix.
        blocks: Blocks that share one heading path.

    Returns:
        Chunks for this heading.
    """
    path = blocks[0].heading_path
    section_path = " > ".join(path)
    kind = _section_kind(path)
    if kind in {"cover", "approval", "boilerplate", "definition", "target"}:
        body = "\n\n".join(block.text for block in blocks)
        pages = _combine_pages(block.pages for block in blocks)
        return _emit(
            document,
            section_path,
            body,
            pages,
            kind,
            kind not in {"cover", "approval", "boilerplate"},
        )
    return _pack_general(document, section_path, _fold(blocks))


def _section_kind(path: tuple[str, ...]) -> str:
    """Classify a heading path that has its own merge rules.

    Args:
        path: Heading path. Empty for text before the first heading.

    Returns:
        ``cover``, ``approval``, ``boilerplate``, ``definition``, ``target``,
        or ``prose`` when the general packer should decide.
    """
    if not path:
        return "cover"
    name = path[-1]
    if name in _APPROVAL:
        return "approval"
    if name in _BOILERPLATE:
        return "boilerplate"
    if name == "Definitions":
        return "definition"
    if "target" in name.casefold():
        return "target"
    return "prose"


def _fold(blocks: list[ParsedBlock]) -> list[_Piece]:
    """Attach units lines and footnotes to tables, and group list items.

    Args:
        blocks: Blocks from one heading path.

    Returns:
        Pieces in reading order.
    """
    pieces: list[_Piece] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if _is_table(block.text) or (_is_units(block.text) and _next_is_table(blocks, index)):
            start = index
            table_at = start + (1 if _is_units(blocks[start].text) else 0)
            end = table_at + 1
            while end < len(blocks) and _is_footnote(blocks[end].text):
                end += 1
            grouped = blocks[start:end]
            pieces.append(
                _Piece(
                    kind="table",
                    body="\n\n".join(item.text for item in grouped),
                    pages=_combine_pages(item.pages for item in grouped),
                )
            )
            index = end
            continue
        if _NUMBERED.match(block.text):
            end = index + 1
            while end < len(blocks) and _BULLET.match(blocks[end].text):
                end += 1
            grouped = blocks[index:end]
            pieces.append(
                _Piece(
                    kind="list",
                    body="\n\n".join(item.text for item in grouped),
                    pages=_combine_pages(item.pages for item in grouped),
                )
            )
            index = end
            continue
        if _BULLET.match(block.text):
            end = index + 1
            while end < len(blocks) and _BULLET.match(blocks[end].text):
                end += 1
            grouped = blocks[index:end]
            pieces.append(
                _Piece(
                    kind="list",
                    body="\n\n".join(item.text for item in grouped),
                    pages=_combine_pages(item.pages for item in grouped),
                )
            )
            index = end
            continue
        pieces.append(_Piece(kind="prose", body=block.text, pages=block.pages))
        index += 1
    return pieces


def _pack_general(
    document: ParsedDocument, section_path: str, pieces: list[_Piece]
) -> list[Chunk]:
    """Merge short prose and keep tables and lists intact.

    Args:
        document: Source document.
        section_path: Heading path for every chunk from these pieces.
        pieces: Folded pieces.

    Returns:
        Chunks for one general heading.
    """
    chunks: list[Chunk] = []
    prose: list[_Piece] = []
    list_intro = ""

    def flush_prose() -> None:
        if not prose:
            return
        body = "\n\n".join(piece.body for piece in prose)
        pages = _combine_pages(piece.pages for piece in prose)
        chunks.extend(_emit(document, section_path, body, pages, "prose", True))
        prose.clear()

    for piece in pieces:
        if piece.kind == "table":
            intro = "\n\n".join(item.body for item in prose)
            if intro and _words(_render(document.title, section_path, f"{intro}\n\n{piece.body}")) <= HARD_MAX:
                prose.clear()
                chunks.extend(
                    _emit(document, section_path, f"{intro}\n\n{piece.body}", piece.pages, "table", True)
                )
            else:
                flush_prose()
                chunks.extend(_emit(document, section_path, piece.body, piece.pages, "table", True, intro))
            continue
        if piece.kind == "list":
            intro = "\n\n".join(item.body for item in prose)
            combined = f"{intro}\n\n{piece.body}".strip() if intro else piece.body
            if intro and _words(intro) < 100 and _words(_render(document.title, section_path, combined)) <= PREFERRED_MAX:
                prose.clear()
                list_intro = intro
                chunks.extend(_emit(document, section_path, combined, piece.pages, "list", True))
            else:
                flush_prose()
                chunks.extend(
                    _emit(document, section_path, piece.body, piece.pages, "list", True, list_intro)
                )
            continue
        prose.append(piece)
    flush_prose()
    return chunks


def _emit(
    document: ParsedDocument,
    section_path: str,
    body: str,
    pages: str,
    content_type: str,
    searchable: bool,
    intro: str = "",
) -> list[Chunk]:
    """Render one body, splitting it when the prefixed text exceeds 500 words.

    Args:
        document: Source document.
        section_path: Heading path placed after the title.
        body: Chunk body. A continuation repeats ``intro`` in the prefix.
        pages: Page string for this body.
        content_type: Stored content type.
        searchable: Whether keyword search should index the chunk.
        intro: List or table introduction repeated on a continuation.

    Returns:
        One chunk, or several when the body must be split.
    """
    parts = _split_body(document.title, section_path, body, intro)
    return [
        _make_chunk(document, section_path, part_body, pages, content_type, searchable, part_intro)
        for part_body, part_intro in parts
    ]


def _split_body(title: str, section_path: str, body: str, intro: str) -> list[tuple[str, str]]:
    """Split a body that does not fit under the hard maximum.

    Args:
        title: Document title.
        section_path: Heading path.
        body: Text to split.
        intro: Introduction already counted in the prefix.

    Returns:
        ``(body, intro)`` pairs. The first pair keeps ``intro`` only when the
        body was not itself a continuation. A body that is one sentence is
        returned whole.
    """
    if _words(_render(title, section_path, body, intro)) <= HARD_MAX:
        return [(body, intro)]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if len(paragraphs) > 1:
        return _pack_parts(title, section_path, paragraphs, intro)
    rows = _split_table_rows(body)
    if len(rows) > 1:
        return [(row, intro if index else "") for index, row in enumerate(rows)]
    sentences = [part.strip() for part in _SENTENCE.split(body) if part.strip()]
    if len(sentences) > 1:
        return _pack_parts(title, section_path, sentences, intro)
    return [(body, intro)]


def _pack_parts(
    title: str, section_path: str, parts: list[str], intro: str
) -> list[tuple[str, str]]:
    """Pack split parts without exceeding the hard maximum.

    Args:
        title: Document title.
        section_path: Heading path.
        parts: Paragraphs or sentences.
        intro: Introduction repeated on each packed piece.

    Returns:
        Packed ``(body, intro)`` pairs.
    """
    packed: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append(("\n\n".join(buffer), intro))
            buffer.clear()

    for part in parts:
        candidate = "\n\n".join([*buffer, part])
        if buffer and _words(_render(title, section_path, candidate, intro)) > HARD_MAX:
            flush()
        buffer.append(part)
    flush()
    return packed


def _split_table_rows(body: str) -> list[str]:
    """Split a markdown table between rows once it is too long.

    Args:
        body: Table text, optionally with a units line and footnotes.

    Returns:
        One string when the body is not a multi-row table. Otherwise each
        string repeats the header row.
    """
    lines = body.splitlines()
    separator = next((index for index, line in enumerate(lines) if re.match(r"^\|\s*---", line)), None)
    if separator is None or separator + 1 >= len(lines):
        return [body]
    header = lines[: separator + 1]
    rest = lines[separator + 1 :]
    rows = [line for line in rest if line.startswith("|")]
    footnotes = [line for line in rest if not line.startswith("|") and line.strip()]
    if len(rows) < 2:
        return [body]
    prelude = "\n".join(line for line in lines[: separator - 1] if not line.startswith("|")).strip()
    pieces: list[str] = []
    bucket: list[str] = []
    for row in rows:
        trial_rows = [*bucket, row]
        trial = "\n".join([*([prelude] if prelude else []), *header, *trial_rows])
        if bucket and _words(trial) > HARD_MAX:
            pieces.append("\n".join([*([prelude] if prelude and not pieces else []), *header, *bucket]))
            bucket = [row]
        else:
            bucket.append(row)
    if bucket:
        tail = [*header, *bucket, *footnotes]
        if prelude and not pieces:
            tail = [prelude, *tail]
        pieces.append("\n".join(tail))
    return pieces or [body]


def _make_chunk(
    document: ParsedDocument,
    section_path: str,
    body: str,
    pages: str,
    content_type: str,
    searchable: bool,
    intro: str,
) -> Chunk:
    """Build one chunk with the repeated prefix.

    Args:
        document: Source document.
        section_path: Heading path.
        body: Chunk body.
        pages: Page string.
        content_type: Stored content type.
        searchable: Whether keyword search should index the chunk.
        intro: Introduction included in the prefix when this is a continuation
            whose body does not already start with that introduction.

    Returns:
        A chunk with an empty id and status.
    """
    prefix_intro = "" if not intro or body.startswith(intro) else intro
    return Chunk(
        chunk_id="",
        text=_render(document.title, section_path, body, prefix_intro),
        document_id=document.document_id,
        document_title=document.title,
        section_path=section_path,
        pages=pages,
        version=document.version,
        content_type=content_type,
        searchable=searchable,
    )


def _render(title: str, section_path: str, body: str, intro: str = "") -> str:
    """Prefix a body with the title, section path, and optional introduction.

    Args:
        title: Document title.
        section_path: Heading path. Omitted when empty.
        body: Chunk body.
        intro: List or table introduction. Omitted when empty.

    Returns:
        The stored chunk text.
    """
    parts = [title]
    if section_path:
        parts.append(section_path)
    if intro:
        parts.append(intro)
    parts.append(body)
    return "\n".join(parts)


def _words(text: str) -> int:
    """Count whitespace-separated words.

    Args:
        text: Chunk or fragment text.

    Returns:
        The word count.
    """
    return len(text.split())


def _combine_pages(pages: Iterable[str]) -> str:
    """Merge page markers into one range.

    Args:
        pages: Page strings such as ``1-1`` and ``2-2``.

    Returns:
        A single page or a ``start-end`` range. Empty when no marker was seen.
    """
    numbers: list[int] = []
    for page in pages:
        numbers.extend(int(part) for part in page.split("-") if part.isdigit())
    if not numbers:
        return ""
    start, end = min(numbers), max(numbers)
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _is_table(text: str) -> bool:
    """Return whether text is a markdown table.

    Args:
        text: Block text.

    Returns:
        True when the block contains a table row and a separator row.
    """
    lines = text.splitlines()
    return any(line.startswith("|") for line in lines) and any(
        re.match(r"^\|\s*---", line) for line in lines
    )


def _is_units(text: str) -> bool:
    """Return whether a line introduces the units for the following table.

    Args:
        text: Block text.

    Returns:
        True for the units lines used in the carbon plan tables.
    """
    return "Unit of Measurement" in text or text.startswith("Emissions (")


def _is_footnote(text: str) -> bool:
    """Return whether a block is a table footnote.

    Args:
        text: Block text.

    Returns:
        True when the block starts with ``*``.
    """
    return text.startswith("*")


def _next_is_table(blocks: list[ParsedBlock], index: int) -> bool:
    """Return whether the next block is a table.

    Args:
        blocks: Blocks in the current heading.
        index: Index of the current block.

    Returns:
        True when a table follows.
    """
    return index + 1 < len(blocks) and _is_table(blocks[index + 1].text)
