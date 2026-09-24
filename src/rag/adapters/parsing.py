"""Parser adapter for policy markdown.

Purpose
-------
Turns one extracted policy file into a document id, title, edition date, and
ordered blocks. A later PDF parser can implement the same interface.

Contents
--------
- ``ParsedBlock``: one paragraph, list, or table under a heading path.
- ``ParsedDocument``: identity plus blocks.
- ``DocumentParser``: the parser interface.
- ``PolicyMarkdownParser``: the extracted markdown files in this repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_PAGE = re.compile(r"<!--\s*source_pages:\s*([^>]+?)\s*-->")
_PUBLICATION = re.compile(r"^Publication date:\s+(.+)$")
_HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_SLUG = re.compile(r"[^a-z0-9]+")
_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


@dataclass(frozen=True)
class ParsedBlock:
    """Text under one heading path, with the page marker that applied to it."""

    text: str
    heading_path: tuple[str, ...]
    pages: str


@dataclass(frozen=True)
class ParsedDocument:
    """One policy edition after parsing, before chunking."""

    document_id: str
    title: str
    version: str
    blocks: tuple[ParsedBlock, ...]


class DocumentParser(Protocol):
    """Reads a file into a ``ParsedDocument``."""

    def parse(self, path: str) -> ParsedDocument:
        """Parse one file.

        Args:
            path: Path to the source file.

        Returns:
            The parsed document.
        """
        ...


class PolicyMarkdownParser:
    """Parser for the extracted policy markdown in this repo.

    The first body line is the title and the document id. ``Publication date``
    becomes ``version``. ``Review Date`` is left in the cover text.
    """

    def parse(self, path: str) -> ParsedDocument:
        """Parse one extracted markdown file.

        Args:
            path: Path to a markdown file.

        Returns:
            A document whose blocks omit page comments and empty headings.

        Raises:
            ValueError: The file has no title line.
        """
        title = ""
        version = ""
        pages = ""
        heading: tuple[str, ...] = ()
        paragraph: list[str] = []
        blocks: list[ParsedBlock] = []

        def flush() -> None:
            text = "\n".join(paragraph).strip()
            paragraph.clear()
            if text:
                blocks.append(ParsedBlock(text=text, heading_path=heading, pages=pages))

        for line in Path(path).read_text(encoding="utf-8").splitlines():
            page_match = _PAGE.search(line)
            if page_match:
                flush()
                pages = page_match.group(1).strip()
                continue
            if line.strip().startswith("<!--"):
                continue
            if line.startswith("# ") and not line.startswith("##"):
                continue
            heading_match = _HEADING.match(line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                name = heading_match.group(2).strip()
                if level == 2:
                    heading = (name,)
                else:
                    heading = (*heading[:1], name)
                continue
            if not line.strip():
                flush()
                continue
            stripped = line.strip()
            if not title:
                title = stripped
            publication = _PUBLICATION.match(stripped)
            if publication:
                version = _iso_date(publication.group(1))
            paragraph.append(stripped)
        flush()
        if not title:
            raise ValueError(f"No title line in {path}")
        return ParsedDocument(
            document_id=_slug(title),
            title=title,
            version=version,
            blocks=tuple(blocks),
        )


def _iso_date(value: str) -> str:
    """Convert a ``28 March 2024`` publication date to ``YYYY-MM-DD``.

    Args:
        value: Date text from a publication line.

    Returns:
        The ISO date.
    """
    day, month, year = value.split()
    return f"{int(year):04d}-{_MONTHS[month]:02d}-{int(day):02d}"


def _slug(title: str) -> str:
    """Turn a title into a stable document id.

    Args:
        title: Document title.

    Returns:
        A lowercase id with words separated by hyphens.
    """
    return _SLUG.sub("-", title.casefold()).strip("-")
