import json
from pathlib import Path

from rag.adapters.base import ChunkRecord
from rag.adapters.keyword_search import ChromaBm25Index


def test_exact_number_ranks_the_chunk_that_contains_it() -> None:
    index = ChromaBm25Index()
    index.upsert(
        [
            _chunk("current-emissions", "current", "India Scope 2 in 2023 was 6,746.70 tCO2e"),
            _chunk("outdated-target", "outdated", "The target is 15% by 2027"),
            _chunk("plastic", "current", "Eliminate single-use plastic by Dec 2026"),
        ]
    )

    hits = index.query("6,746.70", version="current")

    assert hits[0].chunk_id == "current-emissions"
    assert "6,746.70" in hits[0].text
    assert hits[0].score > 0
    assert all(hit.version == "current" for hit in hits)
    assert all(hit.chunk_id != "outdated-target" for hit in hits)


def test_query_keeps_a_policy_code_as_one_token() -> None:
    index = ChromaBm25Index()
    index.upsert(
        [
            _chunk("declaration", "current", "Completed in accordance with PPN 06/21"),
            _chunk("initiatives", "current", "Commit to RE100 for renewable electricity"),
            _chunk("filler", "current", "Cafeteria menus change every quarter"),
        ]
    )

    ppn = index.query("PPN 06/21", version="current")
    re100 = index.query("RE100", version="current")

    assert ppn[0].chunk_id == "declaration"
    assert re100[0].chunk_id == "initiatives"


def test_repeated_chunk_id_replaces_the_old_text() -> None:
    index = ChromaBm25Index()
    index.upsert(
        [
            _chunk("policy::purpose::00", "current", "Water is scarce."),
            _chunk("filler", "current", "Cafeteria menus change every quarter"),
            _chunk("other", "current", "Offices close on public holidays"),
        ]
    )
    index.upsert([_chunk("policy::purpose::00", "current", "Water is a shared resource.")])

    hits = index.query("shared resource", version="current")

    assert len(hits) == 1
    assert hits[0].text == "Water is a shared resource."


def test_upsert_jsonl_reads_chunk_records(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "policy::purpose::00",
                "document_name": "Water-Management-Policy",
                "version": "current",
                "section": "Purpose",
                "source_pages": "3-3",
                "text": "Water is a shared resource.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "chunk_id": "policy::filler::01",
                "document_name": "Water-Management-Policy",
                "version": "current",
                "section": "Other",
                "source_pages": "4-4",
                "text": "Cafeteria menus change every quarter.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "chunk_id": "policy::other::02",
                "document_name": "Water-Management-Policy",
                "version": "current",
                "section": "Hours",
                "source_pages": "5-5",
                "text": "Offices close on public holidays.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = ChromaBm25Index()

    assert index.upsert_jsonl(path) == 3
    hits = index.query("shared resource", version="current")

    assert hits[0].section == "Purpose"
    assert hits[0].document_name == "Water-Management-Policy"


def test_empty_index_returns_no_hits() -> None:
    assert ChromaBm25Index().query("6,746.70") == []


def _chunk(chunk_id: str, version: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version=version,
        section="Current Year Emission: 2023",
        source_pages="2-2",
        text=text,
    )
