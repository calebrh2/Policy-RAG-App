"""Section chunker follows the policy size and metadata rules."""

from __future__ import annotations

from pathlib import Path

from rag.adapters.chunking import SectionChunker
from rag.adapters.parsing import PolicyMarkdownParser

DOCS = Path(__file__).resolve().parents[2] / "data/extracted/RAG-documents"


def test_carbon_plan_chunks_follow_section_rules() -> None:
    document = PolicyMarkdownParser().parse(str(DOCS / "Carbon-Reduction-Plan.md"))
    chunks = SectionChunker().chunk(document)
    initiatives = [chunk for chunk in chunks if chunk.section_path == "Carbon Reduction Initiatives"]

    assert all(len(chunk.text.split()) <= 500 for chunk in chunks)
    assert all(chunk.text.startswith(document.title) for chunk in chunks)
    assert any(chunk.section_path and chunk.text.startswith(f"{document.title}\n{chunk.section_path}") for chunk in chunks)
    assert any("ISO 14001" in chunk.text for chunk in initiatives)
    assert any("Piped Natural Gas" in chunk.text for chunk in initiatives)
    assert not any("ISO 14001" in chunk.text and "Piped Natural Gas" in chunk.text for chunk in initiatives)
    assert any(
        "| Emissions | UK | India |" in chunk.text and "backup diesel generators" in chunk.text
        for chunk in chunks
    )
    needle = "100% conversion of company taxis and buses to Compressed Natural Gas"
    assert sum(needle in chunk.text for chunk in chunks) == 1
    assert any(chunk.content_type == "cover" and not chunk.searchable for chunk in chunks)
    assert any(
        chunk.content_type == "approval" and not chunk.searchable and "Gautam Samanta" in chunk.text
        for chunk in chunks
    )
    assert any(chunk.searchable and "PPN 06/21" in chunk.text for chunk in chunks)
    assert any("Baseline Year: 2017 (India Operations)" in chunk.section_path for chunk in chunks)
    assert any("Baseline Year: 2022 (UK operations)" in chunk.section_path for chunk in chunks)


def test_plastic_policy_marks_boilerplate_and_keeps_subsection_paths() -> None:
    document = PolicyMarkdownParser().parse(str(DOCS / "Single-use-Plastic-free-Policy copy.md"))
    chunks = SectionChunker().chunk(document)

    assert any(
        chunk.content_type == "boilerplate" and not chunk.searchable and "www.coforge.com" in chunk.text
        for chunk in chunks
    )
    assert any(chunk.content_type == "approval" and not chunk.searchable for chunk in chunks)
    assert any(chunk.section_path.endswith("Targets") and " > " in chunk.section_path for chunk in chunks)
    assert any(chunk.section_path.endswith("Monitoring & Reporting") for chunk in chunks)
    assert all(len(chunk.text.split()) <= 500 for chunk in chunks)
