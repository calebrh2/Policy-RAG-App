"""Policy markdown parser reads the extracted corpus."""

from __future__ import annotations

from pathlib import Path

from rag.adapters.parsing import PolicyMarkdownParser

DOCS = Path(__file__).resolve().parents[2] / "data/extracted/RAG-documents"


def test_carbon_plans_share_an_id_and_keep_their_publication_dates() -> None:
    parser = PolicyMarkdownParser()
    current = parser.parse(str(DOCS / "Carbon-Reduction-Plan.md"))
    outdated = parser.parse(str(DOCS / "Outdated-Carbon-Reduction-Plan-TEST-VERSION.md"))

    assert current.document_id == outdated.document_id == "carbon-reduction-plan"
    assert current.title == "Carbon Reduction Plan"
    assert current.version == "2024-03-28"
    assert outdated.version == "2022-09-15"
    assert any(
        block.heading_path
        == ("Baseline Emissions Footprint", "Baseline Year: 2017 (India Operations)")
        for block in current.blocks
    )
    assert all("source_pages" not in block.text for block in current.blocks)


def test_water_policy_has_no_version() -> None:
    document = PolicyMarkdownParser().parse(str(DOCS / "Water-Management-Policy.md"))

    assert document.document_id == "water-management-policy"
    assert document.version == ""
    assert all("source_pages" not in block.text for block in document.blocks)


def test_empty_heading_produces_no_block() -> None:
    document = PolicyMarkdownParser().parse(str(DOCS / "Single-use-Plastic-free-Policy copy.md"))
    goal_blocks = [block for block in document.blocks if block.heading_path == ("Goal",)]

    assert goal_blocks
    assert all(block.text for block in goal_blocks)
    assert any(block.heading_path == ("Elimination & Substitution (Mandatory Prohibition)",) for block in document.blocks)
