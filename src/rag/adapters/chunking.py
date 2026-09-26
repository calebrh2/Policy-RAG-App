"""Section-first chunker for parsed policy documents.

Purpose
-------
Splits a ``ParsedDocument`` on markdown heading paths. The first pass is a
whitespace word count. A rendered chunk over 512 tokens is measured with the
``BAAI/bge-small-en-v1.5`` tokenizer.

Contents
--------
- ``Chunk``: one stored piece of a document, before ingest assigns status.
- ``Chunker``: the chunker interface.
- ``SectionChunker``: the policy section rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rag.adapters.parsing import ParsedBlock, ParsedDocument

PREFERRED_MAX = 450
HARD_MAX = 500
TOKEN_MAX = 512
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
    500 words, including the repeated title and section path. A paragraph over
    that maximum splits on sentences. A rendered chunk over 512 tokens for
    ``BAAI/bge-small-en-v1.5`` splits on table rows, then sentences, then
    words. A single word longer than 512 tokens stays intact.
    """

    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        """Store the counter used for the 512-token cap.

        Args:
            token_counter: Returns the token length of a rendered chunk.
                Defaults to the ``BAAI/bge-small-en-v1.5`` tokenizer.
        """
        self._token_counter = token_counter or _default_token_count

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk one parsed document.

        Args:
            document: Parser output.

        Returns:
            Chunks in reading order.
        """
        chunks: list[Chunk] = []
        for blocks in _group_by_heading(document.blocks):
            chunks.extend(_chunks_for_heading(document, blocks, self._token_counter))
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


def _chunks_for_heading(
    document: ParsedDocument,
    blocks: list[ParsedBlock],
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Pack one heading path into chunks.

    Args:
        document: Source document, used for the title prefix.
        blocks: Blocks that share one heading path.
        count_tokens: Token length of a rendered chunk.

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
            count_tokens=count_tokens,
        )
    return _pack_general(document, section_path, _fold(blocks), count_tokens)


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
    document: ParsedDocument,
    section_path: str,
    pieces: list[_Piece],
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Merge short prose and keep tables and lists intact.

    Args:
        document: Source document.
        section_path: Heading path for every chunk from these pieces.
        pieces: Folded pieces.
        count_tokens: Token length of a rendered chunk.

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
        chunks.extend(
            _emit(document, section_path, body, pages, "prose", True, count_tokens=count_tokens)
        )
        prose.clear()

    for piece in pieces:
        if piece.kind == "table":
            intro = "\n\n".join(item.body for item in prose)
            if intro and _words(_render(document.title, section_path, f"{intro}\n\n{piece.body}")) <= HARD_MAX:
                prose.clear()
                chunks.extend(
                    _emit(
                        document,
                        section_path,
                        f"{intro}\n\n{piece.body}",
                        piece.pages,
                        "table",
                        True,
                        count_tokens=count_tokens,
                    )
                )
            else:
                flush_prose()
                chunks.extend(
                    _emit(
                        document,
                        section_path,
                        piece.body,
                        piece.pages,
                        "table",
                        True,
                        intro,
                        count_tokens,
                    )
                )
            continue
        if piece.kind == "list":
            intro = "\n\n".join(item.body for item in prose)
            combined = f"{intro}\n\n{piece.body}".strip() if intro else piece.body
            if intro and _words(intro) < 100 and _words(_render(document.title, section_path, combined)) <= PREFERRED_MAX:
                prose.clear()
                list_intro = intro
                chunks.extend(
                    _emit(
                        document,
                        section_path,
                        combined,
                        piece.pages,
                        "list",
                        True,
                        count_tokens=count_tokens,
                    )
                )
            else:
                flush_prose()
                chunks.extend(
                    _emit(
                        document,
                        section_path,
                        piece.body,
                        piece.pages,
                        "list",
                        True,
                        list_intro,
                        count_tokens,
                    )
                )
            continue
        prose.append(piece)
    flush_prose()
    return chunks


_bge_tokenizer: PreTrainedTokenizerBase | None = None


def _default_token_count(text: str) -> int:
    """Count tokens with the BGE tokenizer once the text can exceed 512.

    Args:
        text: Rendered chunk text.

    Returns:
        An upper bound when the string is at most 510 characters, otherwise
        the tokenizer length including ``[CLS]`` and ``[SEP]``.
    """
    if len(text) <= TOKEN_MAX - 2:
        return len(text) + 2
    global _bge_tokenizer
    if _bge_tokenizer is None:
        from transformers import AutoTokenizer
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        from rag.adapters.embedding import DEFAULT_DENSE_MODEL

        loaded = AutoTokenizer.from_pretrained(DEFAULT_DENSE_MODEL)  # type: ignore[no-untyped-call]
        _bge_tokenizer = cast(PreTrainedTokenizerBase, loaded)
    return len(_bge_tokenizer.encode(text, add_special_tokens=True, truncation=False))


def _emit(
    document: ParsedDocument,
    section_path: str,
    body: str,
    pages: str,
    content_type: str,
    searchable: bool,
    intro: str = "",
    count_tokens: Callable[[str], int] = _default_token_count,
) -> list[Chunk]:
    """Render one body, splitting it when it exceeds a size limit.

    Word packing runs first. A rendered chunk over 512 tokens is split after that.

    Args:
        document: Source document.
        section_path: Heading path placed after the title.
        body: Chunk body. A continuation repeats ``intro`` in the prefix.
        pages: Page string for this body.
        content_type: Stored content type.
        searchable: Whether keyword search should index the chunk.
        intro: List or table introduction repeated on a continuation.
        count_tokens: Token length of a rendered chunk.

    Returns:
        One chunk, or several when the body must be split.
    """
    parts = _split_body(document.title, section_path, body, intro)
    fitted: list[tuple[str, str]] = []
    for part_body, part_intro in parts:
        fitted.extend(
            _fit_token_limit(document.title, section_path, part_body, part_intro, count_tokens)
        )
    return [
        _make_chunk(document, section_path, part_body, pages, content_type, searchable, part_intro)
        for part_body, part_intro in fitted
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
        body was not itself a continuation. A paragraph over the hard maximum
        splits on sentences. A body that is one sentence is returned whole.
    """
    if _words(_render(title, section_path, body, intro)) <= HARD_MAX:
        return [(body, intro)]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if len(paragraphs) > 1:
        expanded = _expand_long_paragraphs(title, section_path, paragraphs, intro)
        return _pack_parts(title, section_path, expanded, intro)
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


def _expand_long_paragraphs(
    title: str, section_path: str, paragraphs: list[str], intro: str
) -> list[str]:
    """Split only paragraphs whose rendered text exceeds the hard maximum.

    Args:
        title: Document title.
        section_path: Heading path.
        paragraphs: Paragraphs from one body, in order.
        intro: Introduction counted in the rendered prefix.

    Returns:
        The same paragraphs, with each oversized paragraph replaced by its
        sentences. A paragraph of one sentence stays whole.
    """
    expanded: list[str] = []
    for paragraph in paragraphs:
        if _words(_render(title, section_path, paragraph, intro)) <= HARD_MAX:
            expanded.append(paragraph)
            continue
        sentences = [part.strip() for part in _SENTENCE.split(paragraph) if part.strip()]
        if len(sentences) > 1:
            expanded.extend(sentences)
        else:
            expanded.append(paragraph)
    return expanded


def _fit_token_limit(
    title: str,
    section_path: str,
    body: str,
    intro: str,
    count_tokens: Callable[[str], int],
) -> list[tuple[str, str]]:
    """Split a rendered chunk that exceeds 512 tokens.

    Args:
        title: Document title.
        section_path: Heading path.
        body: Chunk body.
        intro: Introduction repeated on a continuation.
        count_tokens: Token length of a rendered chunk.

    Returns:
        ``(body, intro)`` pairs. A body at or under 512 tokens is unchanged.
        A table splits between rows before a row is cut. A single word that
        still exceeds the cap stays intact.
    """
    if not _over_token_limit(title, section_path, body, intro, count_tokens):
        return [(body, intro)]
    rows = _split_table_rows(
        body,
        lambda text: _over_token_limit(title, section_path, text, intro, count_tokens),
    )
    if len(rows) > 1:
        fitted: list[tuple[str, str]] = []
        for index, row in enumerate(rows):
            row_intro = intro if index else ""
            fitted.extend(_fit_token_limit(title, section_path, row, row_intro, count_tokens))
        return fitted
    return _fit_sentences_then_words(title, section_path, body, intro, count_tokens)


def _fit_sentences_then_words(
    title: str,
    section_path: str,
    body: str,
    intro: str,
    count_tokens: Callable[[str], int],
) -> list[tuple[str, str]]:
    """Split an oversized body on sentences, then on words.

    Args:
        title: Document title.
        section_path: Heading path.
        body: Chunk body that is already over the token cap.
        intro: Introduction repeated on a continuation.
        count_tokens: Token length of a rendered chunk.

    Returns:
        Packed ``(body, intro)`` pairs. One word over the cap is returned whole.
    """
    if not _over_token_limit(title, section_path, body, intro, count_tokens):
        return [(body, intro)]
    sentences = [part.strip() for part in _SENTENCE.split(body) if part.strip()]
    if len(sentences) > 1:
        return _pack_token_parts(
            title, section_path, sentences, intro, count_tokens, "\n\n", split_words=True
        )
    return _fit_words(title, section_path, body, intro, count_tokens)


def _fit_words(
    title: str,
    section_path: str,
    body: str,
    intro: str,
    count_tokens: Callable[[str], int],
) -> list[tuple[str, str]]:
    """Split an oversized sentence on whitespace.

    Args:
        title: Document title.
        section_path: Heading path.
        body: One sentence, or text with no sentence boundary.
        intro: Introduction repeated on a continuation.
        count_tokens: Token length of a rendered chunk.

    Returns:
        Packed ``(body, intro)`` pairs. One word over the cap is returned whole.
    """
    if not _over_token_limit(title, section_path, body, intro, count_tokens):
        return [(body, intro)]
    words = body.split()
    if len(words) > 1:
        return _pack_token_parts(
            title, section_path, words, intro, count_tokens, " ", split_words=False
        )
    return [(body, intro)]


def _pack_token_parts(
    title: str,
    section_path: str,
    parts: list[str],
    intro: str,
    count_tokens: Callable[[str], int],
    joiner: str,
    split_words: bool,
) -> list[tuple[str, str]]:
    """Pack parts until the next one would exceed 512 tokens.

    Args:
        title: Document title.
        section_path: Heading path.
        parts: Sentences or words.
        intro: Introduction repeated on each packed piece.
        count_tokens: Token length of a rendered chunk.
        joiner: String placed between packed parts.
        split_words: When true, a part that is itself over the cap splits on words.

    Returns:
        Packed ``(body, intro)`` pairs.
    """
    packed: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append((joiner.join(buffer), intro))
            buffer.clear()

    for part in parts:
        candidate = joiner.join([*buffer, part])
        if buffer and _over_token_limit(title, section_path, candidate, intro, count_tokens):
            flush()
        if (
            not buffer
            and split_words
            and _over_token_limit(title, section_path, part, intro, count_tokens)
        ):
            packed.extend(_fit_words(title, section_path, part, intro, count_tokens))
            continue
        buffer.append(part)
    flush()
    return packed


def _over_token_limit(
    title: str,
    section_path: str,
    body: str,
    intro: str,
    count_tokens: Callable[[str], int],
) -> bool:
    """Return whether the rendered chunk is over 512 tokens.

    Args:
        title: Document title.
        section_path: Heading path.
        body: Chunk body.
        intro: Introduction included in the prefix.
        count_tokens: Token length of a rendered chunk.

    Returns:
        True when the rendered text exceeds ``TOKEN_MAX``.
    """
    return count_tokens(_render(title, section_path, body, intro)) > TOKEN_MAX


def _split_table_rows(
    body: str, exceeds: Callable[[str], bool] | None = None
) -> list[str]:
    """Split a markdown table between rows once it is too long.

    Args:
        body: Table text, optionally with a units line and footnotes.
        exceeds: Returns whether one table piece is too long. Defaults to the
            500-word hard maximum.

    Returns:
        One string when the body is not a multi-row table. Otherwise each
        string repeats the header row.
    """
    too_long = exceeds or (lambda text: _words(text) > HARD_MAX)
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
        if bucket and too_long(trial):
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
