"""Adaptive, structure-aware chunking for cleaned policy Markdown."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

SOURCE_FILE_RE = re.compile(r"<!--\s*source_file:\s*(.*?)\s*-->")
SOURCE_PAGES_RE = re.compile(r"<!--\s*source_pages:\s*(\d+)-(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NUMBERED_RE = re.compile(r"^\d+\.\s+")
NON_SEARCHABLE = {
    "Declaration and Sign Off",
    "Signed on behalf of the Supplier:",
    "Approved by:",
    "About Coforge",
}


@dataclass(frozen=True)
class MarkdownBlock:
    kind: Literal["paragraph", "list_item", "table"]
    text: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class MarkdownSection:
    document_title: str
    source_file: str
    heading_path: tuple[str, ...]
    blocks: tuple[MarkdownBlock, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    source_file: str
    version: str
    status: Literal["current", "superseded"]
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    content_types: tuple[str, ...]
    searchable: bool
    estimated_tokens: int
    text: str
    retrieval_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    version: str
    status: Literal["current", "superseded"]


DOCUMENT_VERSIONS = {
    "Carbon-Reduction-Plan.md": DocumentVersion("carbon-reduction-plan", "2024-03-28", "current"),
    "Outdated-Carbon-Reduction-Plan-TEST-VERSION.md": DocumentVersion(
        "carbon-reduction-plan", "2022-09-15", "superseded"
    ),
    "Single-use-Plastic-free-Policy copy.md": DocumentVersion(
        "single-use-plastic-free-policy", "2026-03-10", "current"
    ),
    "Water-Management-Policy.md": DocumentVersion(
        "water-management-policy", "2026-08-10", "current"
    ),
}


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _block_kind(text: str) -> Literal["paragraph", "list_item", "table"]:
    if text.lstrip().startswith("|"):
        return "table"
    if re.match(r"^(?:[-*+]\s+|\d+\.\s+)", text):
        return "list_item"
    return "paragraph"


def parse_markdown_sections(path: Path) -> list[MarkdownSection]:
    source_file, title = path.with_suffix(".pdf").name, path.stem
    headings: list[str] = []
    pages = (1, 1)
    sections: list[MarkdownSection] = []
    blocks: list[MarkdownBlock] = []

    def flush() -> None:
        nonlocal blocks
        if blocks:
            sections.append(MarkdownSection(title, source_file, tuple(headings), tuple(blocks)))
            blocks = []

    for raw in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        unit = raw.strip()
        if not unit:
            continue
        if match := SOURCE_FILE_RE.fullmatch(unit):
            source_file = match.group(1)
        elif match := SOURCE_PAGES_RE.fullmatch(unit):
            pages = (int(match.group(1)), int(match.group(2)))
        elif match := HEADING_RE.fullmatch(unit):
            level = len(match.group(1))
            if level == 1:
                title = match.group(2).strip()
                continue
            flush()
            headings = headings[: level - 2]
            headings.append(match.group(2).strip())
        else:
            blocks.append(MarkdownBlock(_block_kind(unit), unit, *pages))
    flush()
    return sections


def _merge_parent_intros(
    sections: list[MarkdownSection], preferred_min: int
) -> list[MarkdownSection]:
    result: list[MarkdownSection] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        following = sections[index + 1] if index + 1 < len(sections) else None
        size = sum(estimate_tokens(block.text) for block in section.blocks)
        if (
            size < preferred_min
            and following
            and len(following.heading_path) == len(section.heading_path) + 1
            and following.heading_path[:-1] == section.heading_path
        ):
            result.append(replace(following, blocks=section.blocks + following.blocks))
            index += 2
        else:
            result.append(section)
            index += 1
    return result


def _semantic_units(blocks: tuple[MarkdownBlock, ...]) -> list[list[MarkdownBlock]]:
    """Keep every numbered item with all of its following bullets."""
    units: list[list[MarkdownBlock]] = []
    current: list[MarkdownBlock] = []
    numbered = False
    for block in blocks:
        starts_number = block.kind == "list_item" and bool(NUMBERED_RE.match(block.text))
        bullet = block.kind == "list_item" and block.text.startswith(("-", "*", "+"))
        if starts_number:
            if current:
                units.append(current)
            current, numbered = [block], True
        elif numbered and bullet:
            current.append(block)
        else:
            if current:
                units.append(current)
            current, numbered = [block], False
    if current:
        units.append(current)
    return units


def _pack_large_section(
    blocks: tuple[MarkdownBlock, ...], preferred_min: int, target: int, hard_max: int
) -> list[list[MarkdownBlock]]:
    groups: list[list[MarkdownBlock]] = []
    current: list[MarkdownBlock] = []
    size = 0
    for unit in _semantic_units(blocks):
        unit_size = sum(estimate_tokens(block.text) for block in unit)
        if unit_size > hard_max:
            raise ValueError("Atomic paragraph, numbered item, or table exceeds hard maximum")
        if current and size + unit_size > target:
            groups.append(current)
            current, size = [], 0
        current.extend(unit)
        size += unit_size
    if current:
        groups.append(current)
    last_size = sum(estimate_tokens(block.text) for block in groups[-1])
    if len(groups) > 1 and last_size < preferred_min:
        combined = sum(estimate_tokens(block.text) for group in groups[-2:] for block in group)
        if combined <= hard_max:
            groups[-2].extend(groups.pop())
    return groups


def _table_as_text(markdown: str) -> str:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in markdown.splitlines()
        if line.strip() and not re.fullmatch(r"\|?[\s|:-]+\|?", line)
    ]
    if len(rows) < 2:
        return markdown
    headers = rows[0]
    output = [f"Table columns: {', '.join(headers)}."]
    for row in rows[1:]:
        values = [
            f"{header} = {value}"
            for header, value in zip(headers[1:], row[1:], strict=False)
            if value
        ]
        output.append(f"{row[0]}: {', '.join(values)}.")
    return " ".join(output)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:60] or "document"


def create_chunks(
    path: Path,
    *,
    preferred_min_tokens: int = 100,
    target_tokens: int = 350,
    soft_max_tokens: int = 450,
    hard_max_tokens: int = 500,
) -> list[Chunk]:
    if not 0 < preferred_min_tokens <= target_tokens <= soft_max_tokens <= hard_max_tokens:
        raise ValueError("Expected minimum <= target <= soft maximum <= hard maximum")
    version = DOCUMENT_VERSIONS.get(
        path.name, DocumentVersion(_slug(path.stem), "unknown", "current")
    )
    sections = _merge_parent_intros(parse_markdown_sections(path), preferred_min_tokens)
    chunks: list[Chunk] = []
    for section in sections:
        total = sum(estimate_tokens(block.text) for block in section.blocks)
        groups = (
            [list(section.blocks)]
            if total <= soft_max_tokens
            else _pack_large_section(
                section.blocks, preferred_min_tokens, target_tokens, hard_max_tokens
            )
        )
        name = section.heading_path[-1] if section.heading_path else "metadata"
        searchable = bool(section.heading_path) and name not in NON_SEARCHABLE
        for part, group in enumerate(groups, 1):
            text = "\n\n".join(block.text for block in group)
            retrieval_text = "\n".join(
                [
                    f"Document: {section.document_title}.",
                    f"Section: {' > '.join(section.heading_path) or 'Document metadata'}.",
                    *[
                        _table_as_text(block.text) if block.kind == "table" else block.text
                        for block in group
                    ],
                ]
            )
            digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{version.document_id}:{version.version}:{_slug(name)}:{part:02d}:{digest}"
                    ),
                    document_id=version.document_id,
                    document_title=section.document_title,
                    source_file=section.source_file,
                    version=version.version,
                    status=version.status,
                    section_path=section.heading_path,
                    page_start=min(block.page_start for block in group),
                    page_end=max(block.page_end for block in group),
                    content_types=tuple(dict.fromkeys(block.kind for block in group)),
                    searchable=searchable,
                    estimated_tokens=estimate_tokens(retrieval_text),
                    text=text,
                    retrieval_text=retrieval_text,
                )
            )
    return chunks


FirstPassChunk = Chunk


def create_first_pass_chunks(path: Path, *, max_estimated_tokens: int = 400) -> list[Chunk]:
    return create_chunks(
        path,
        preferred_min_tokens=min(100, max_estimated_tokens),
        target_tokens=max_estimated_tokens,
        soft_max_tokens=max_estimated_tokens,
        hard_max_tokens=max_estimated_tokens,
    )
