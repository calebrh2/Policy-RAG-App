from rag.chunking import chunk_markdown, count_tokens


def test_heading_becomes_one_chunk() -> None:
    markdown = """# Policy

<!-- source_pages: 3-3 -->

## Purpose

Water is a shared resource. Coforge will manage it carefully across every site.
"""
    chunks = chunk_markdown(markdown, "Water-Management-Policy")

    assert len(chunks) == 1
    assert chunks[0].section == "Purpose"
    assert chunks[0].version == "current"
    assert chunks[0].source_pages == "3-3"
    assert chunks[0].text.startswith("Water-Management-Policy > Purpose")
    assert "shared resource" in chunks[0].text


def test_empty_heading_becomes_the_parent_of_the_next_chunk() -> None:
    markdown = """## Targets & KPIs

### Targets

Eliminate prohibited items at every India site by Dec 2026.
"""
    chunks = chunk_markdown(markdown, "Single-use-Plastic-free-Policy copy")

    assert len(chunks) == 1
    assert chunks[0].section == "Targets & KPIs > Targets"


def test_about_coforge_is_dropped_and_a_short_signoff_joins_the_previous_section() -> None:
    markdown = """## Compliance Obligations

The ESG Committee may amend this Policy with Board approval and will record the change.

## Approved by:

John Speight

## About Coforge

Coforge is a global digital services provider with offices in many countries.
"""
    chunks = chunk_markdown(markdown, "Water-Management-Policy")

    assert len(chunks) == 1
    assert "John Speight" in chunks[0].text
    assert "global digital services" not in chunks[0].text


def test_long_numbered_section_splits_without_copying_the_previous_item() -> None:
    item = "This initiative covers transport, buildings, and energy use. " * 40
    markdown = f"""## Carbon Reduction Initiatives

Introduced across global operations.

1. {item}

2. {item}
"""
    chunks = chunk_markdown(markdown, "Carbon-Reduction-Plan")

    assert count_tokens(item) > 200
    assert [chunk.section for chunk in chunks] == [
        "Carbon Reduction Initiatives > 1",
        "Carbon Reduction Initiatives > 2",
    ]
    assert "1. " in chunks[0].text
    assert "2. " not in chunks[0].text
    assert chunks[0].version == "current"


def test_outdated_filename_is_marked_outdated() -> None:
    markdown = "## Commitment to achieving Net Zero\n\nNet zero by 2050 remains the target."
    chunks = chunk_markdown(markdown, "Outdated-Carbon-Reduction-Plan-TEST-VERSION")

    assert chunks[0].version == "outdated"
