"""Structural chunks for the policy Markdown files.

One heading is one chunk. A section is split only when it is over 512 tokens.
Numbered items and subsections do not overlap. A chunk under 40 tokens joins
the previous section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_TOKENS = 512
MIN_TOKENS = 40

_HEADING = re.compile(r"^(#{2,3}) +(.+?)\s*$")
_PAGE = re.compile(r"^<!-- source_pages:\s*([^>]+?)\s*-->$")
_NUMBERED = re.compile(r"^(\d+)\. ")
_BULLET = re.compile(r"^[-*•] ")
_SPAN = re.compile(r"(\d+)\s*-\s*(\d+)")


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a policy document."""

    chunk_id: str
    document_name: str
    version: str
    section: str
    source_pages: str
    text: str
    token_count: int


@dataclass
class _Section:
    title: str
    level: int
    parents: tuple[str, ...]
    body: str
    pages: tuple[tuple[int, int], ...]


def count_tokens(text: str) -> int:
    """Estimate tokens at about four characters each, enough to apply the 512 cutoff."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def version_for(document_name: str) -> str:
    """Mark the planted outdated carbon plan. Every other file is current."""
    lowered = document_name.casefold()
    if "outdated" in lowered or "test-version" in lowered:
        return "outdated"
    return "current"


def chunk_markdown(markdown: str, document_name: str) -> list[Chunk]:
    """Turn one Markdown document into structural chunks."""
    sections = _sections(markdown)
    pieces: list[tuple[str, str, str]] = []
    for section in sections:
        if section.title == "About Coforge":
            continue
        pieces.extend(_pieces(section))

    merged = _merge_short(pieces)
    version = version_for(document_name)
    chunks: list[Chunk] = []
    for index, (section_name, pages, body) in enumerate(merged):
        text = f"{document_name} > {section_name}\n\n{body.strip()}"
        slug = _slug(section_name)
        chunks.append(
            Chunk(
                chunk_id=f"{document_name}.md::{slug}::{index:02d}",
                document_name=document_name,
                version=version,
                section=section_name,
                source_pages=pages,
                text=text,
                token_count=count_tokens(text),
            )
        )
    return chunks


def chunk_directory(directory: Path) -> list[Chunk]:
    """Chunk every Markdown file in a directory, in filename order."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.stem))
    return chunks


def _sections(markdown: str) -> list[_Section]:
    preamble: list[str] = []
    sections: list[_Section] = []
    current_title = ""
    current_level = 0
    current_parents: tuple[str, ...] = ()
    current_lines: list[str] = []
    current_pages: list[tuple[int, int]] = []
    seen_heading = False
    heading_stack: list[tuple[int, str]] = []
    pending_parents: list[str] = []
    pending_pages: list[tuple[int, int]] = []

    def flush() -> None:
        if not seen_heading or not current_title:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            pending_parents.append(current_title)
            return
        parents = tuple(dict.fromkeys(pending_parents + list(current_parents)))
        pending_parents.clear()
        sections.append(
            _Section(current_title, current_level, parents, body, tuple(current_pages))
        )

    for line in markdown.splitlines():
        page = _PAGE.match(line.strip())
        if page:
            pending_pages.extend(_spans(page.group(1)))
            continue
        if line.startswith("<!--"):
            continue
        heading = _HEADING.match(line)
        if heading:
            inherited = current_pages[-1:] if current_pages else []
            flush()
            seen_heading = True
            level = len(heading.group(1))
            current_title = heading.group(2).strip()
            current_level = level
            heading_stack[:] = [item for item in heading_stack if item[0] < level]
            current_parents = tuple(title for _, title in heading_stack)
            heading_stack.append((level, current_title))
            current_lines = []
            current_pages = pending_pages or inherited
            pending_pages = []
            continue
        if not line.strip():
            if seen_heading:
                current_lines.append(line)
            continue
        if pending_pages and seen_heading:
            current_pages.extend(pending_pages)
            pending_pages = []
        if seen_heading:
            current_lines.append(line)
        elif line.strip():
            preamble.append(line)

    flush()
    if preamble and sections:
        intro = "\n".join(preamble).strip()
        first = sections[0]
        sections[0] = _Section(
            first.title,
            first.level,
            first.parents,
            f"{intro}\n\n{first.body}",
            first.pages,
        )
    return sections


def _pieces(section: _Section) -> list[tuple[str, str, str]]:
    path = _section_path(section)
    pages = _format_pages(section.pages)
    body = section.body.strip()
    if count_tokens(f"{path}\n\n{body}") <= MAX_TOKENS:
        return [(path, pages, body)]

    numbered = _split_marked(body, _NUMBERED)
    if len(numbered) > 1:
        return _fit(path, pages, numbered)
    bulleted = _split_marked(body, _BULLET)
    if len(bulleted) > 1:
        return _fit(path, pages, bulleted)
    return [(path, pages, body)]


def _fit(path: str, pages: str, parts: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    fitted: list[tuple[str, str, str]] = []
    for label, part in parts:
        section = f"{path} > {label}" if label else path
        if count_tokens(f"{section}\n\n{part}") <= MAX_TOKENS or _BULLET.search(part) is None:
            fitted.append((section, pages, part))
            continue
        smaller = _split_marked(part, _BULLET)
        if len(smaller) <= 1:
            fitted.append((section, pages, part))
            continue
        fitted.extend(_fit(section, pages, smaller))
    return fitted


def _split_marked(body: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    lines = body.splitlines()
    starts = [index for index, line in enumerate(lines) if pattern.match(line)]
    if len(starts) < 2:
        return [("", body.strip())]

    parts: list[tuple[str, str]] = []
    intro = "\n".join(lines[: starts[0]]).strip()
    for position, start in enumerate(starts):
        next_start = starts[position + 1] if position + 1 < len(starts) else len(lines)
        end = _item_end(lines, start, next_start, position == len(starts) - 1)
        block = "\n".join(lines[start:end]).strip()
        match = pattern.match(lines[start])
        label = match.group(1) if match and match.lastindex else ""
        if position == 0 and intro:
            block = f"{intro}\n\n{block}"
        parts.append((label, block))
        if position == len(starts) - 1 and end < len(lines):
            trailing = "\n".join(lines[end:]).strip()
            if trailing:
                parts.append(("", trailing))
    return [(label, text) for label, text in parts if text.strip()]


def _item_end(lines: list[str], start: int, next_start: int, last: bool) -> int:
    """Keep trailing prose after the final list item out of that item."""
    if not last:
        return next_start
    bullet_ends = [index for index in range(start, len(lines)) if _BULLET.match(lines[index])]
    if not bullet_ends:
        return len(lines)
    end = bullet_ends[-1] + 1
    while end < len(lines) and not lines[end].strip():
        end += 1
    return end


def _merge_short(pieces: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    merged: list[tuple[str, str, str]] = []
    for section, pages, body in pieces:
        if merged and count_tokens(body) < MIN_TOKENS:
            previous_section, previous_pages, previous_body = merged[-1]
            combined = f"{previous_body.rstrip()}\n\n{section}\n\n{body.strip()}"
            merged[-1] = (previous_section, _join_pages(previous_pages, pages), combined)
            continue
        merged.append((section, pages, body))
    return merged


def _section_path(section: _Section) -> str:
    titles = [title for title in section.parents if title != section.title]
    titles.append(section.title)
    return " > ".join(titles)


def _spans(label: str) -> list[tuple[int, int]]:
    found = [(int(start), int(end)) for start, end in _SPAN.findall(label)]
    return found


def _format_pages(pages: tuple[tuple[int, int], ...]) -> str:
    if not pages:
        return ""
    start = min(page[0] for page in pages)
    end = max(page[1] for page in pages)
    return f"{start}-{end}"


def _join_pages(left: str, right: str) -> str:
    spans = _spans(left) + _spans(right)
    return _format_pages(tuple(spans))


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug[:80] or "section"
