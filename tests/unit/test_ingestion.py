from pathlib import Path

from rag.ingestion import create_first_pass_chunks, parse_markdown_sections


def test_parser_preserves_section_paths_pages_and_tables(tmp_path: Path) -> None:
    source = tmp_path / "Policy.md"
    source.write_text(
        """# Policy

<!-- source_file: Policy.pdf -->

<!-- source_pages: 2-2 -->

## Targets

Target context.

### Metrics

<!-- source_pages: 3-4 -->

| Metric | Value |
| --- | --- |
| Reduction | 20% |
""",
        encoding="utf-8",
    )
    sections = parse_markdown_sections(source)
    assert [section.heading_path for section in sections] == [
        ("Targets",),
        ("Targets", "Metrics"),
    ]
    assert sections[1].blocks[0].kind == "table"
    assert (sections[1].blocks[0].page_start, sections[1].blocks[0].page_end) == (3, 4)


def test_first_pass_splits_only_between_atomic_blocks(tmp_path: Path) -> None:
    source = tmp_path / "Policy.md"
    source.write_text(
        """# Policy

## Rules

First complete paragraph with several words.

Second complete paragraph with several words.
""",
        encoding="utf-8",
    )
    chunks = create_first_pass_chunks(source, max_estimated_tokens=8)
    assert len(chunks) == 2
    assert chunks[0].text == "First complete paragraph with several words."
    assert chunks[1].text == "Second complete paragraph with several words."
    assert all(chunk.section_path == ("Rules",) for chunk in chunks)
