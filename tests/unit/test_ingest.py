"""Ingest keeps Carbon Plan editions separate and drops stale chunk ids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rag.ingest import ingest
from rag.keyword_index import KeywordIndex

DOCS = Path(__file__).resolve().parents[2] / "data/extracted/RAG-documents"


class FakeCollection:
    """In-memory stand-in for the Chroma collection methods ingest calls."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, str | bool]],
    ) -> None:
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.rows[chunk_id] = {"document": document, "metadata": dict(metadata)}

    def get(self, *, where: Mapping[str, object]) -> Mapping[str, object]:
        clauses = where.get("$and")
        filters = (
            {key: value for clause in clauses for key, value in clause.items()}
            if isinstance(clauses, list)
            else where
        )
        matched = [
            (chunk_id, row)
            for chunk_id, row in self.rows.items()
            if all(row["metadata"][key] == value for key, value in filters.items())  # type: ignore[index]
        ]
        return {
            "ids": [chunk_id for chunk_id, _ in matched],
            "metadatas": [row["metadata"] for _, row in matched],
        }

    def delete(self, *, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self.rows.pop(chunk_id, None)

    def update(
        self,
        *,
        ids: Sequence[str],
        metadatas: Sequence[Mapping[str, str | bool]],
    ) -> None:
        for chunk_id, metadata in zip(ids, metadatas, strict=True):
            self.rows[chunk_id]["metadata"] = dict(metadata)


def test_carbon_editions_stay_separate_and_search_defaults_to_current(tmp_path: Path) -> None:
    index = KeywordIndex(str(tmp_path / "chunks.sqlite"))
    collection = FakeCollection()
    ingest(
        [
            str(DOCS / "Carbon-Reduction-Plan.md"),
            str(DOCS / "Outdated-Carbon-Reduction-Plan-TEST-VERSION.md"),
        ],
        index,
        collection,
    )

    current = [
        row["metadata"]
        for row in collection.rows.values()
        if row["metadata"]["status"] == "current"  # type: ignore[index]
    ]
    outdated = [
        row["metadata"]
        for row in collection.rows.values()
        if row["metadata"]["status"] == "outdated"  # type: ignore[index]
    ]
    assert current
    assert outdated
    assert {metadata["version"] for metadata in current} == {"2024-03-28"}
    assert {metadata["version"] for metadata in outdated} == {"2022-09-15"}
    assert index.search("15") == []
    assert index.search("15", current_only=False)[0].chunk_id.startswith("carbon-reduction-plan:2022-09-15:")
    covers = [
        row
        for row in collection.rows.values()
        if row["metadata"]["content_type"] == "cover"  # type: ignore[index]
    ]
    assert covers
    assert index.search("Supplier") == []
    index.close()


def test_reingest_deletes_missing_chunk_ids_and_leaves_the_other_edition(tmp_path: Path) -> None:
    index = KeywordIndex(str(tmp_path / "chunks.sqlite"))
    collection = FakeCollection()
    paths = [
        str(DOCS / "Carbon-Reduction-Plan.md"),
        str(DOCS / "Outdated-Carbon-Reduction-Plan-TEST-VERSION.md"),
    ]
    ingest(paths, index, collection)
    before = {
        chunk_id
        for chunk_id in collection.rows
        if chunk_id.startswith("carbon-reduction-plan:2024-03-28:")
    }
    short = tmp_path / "short.md"
    short.write_text(
        "# Carbon-Reduction-Plan\n\n"
        "Carbon Reduction Plan\n\n"
        "Publication date: 28 March 2024\n\n"
        "## Only Section\n\n"
        "One short rule about emissions.\n",
        encoding="utf-8",
    )

    ingest([str(short), paths[1]], index, collection)

    after = {
        chunk_id
        for chunk_id in collection.rows
        if chunk_id.startswith("carbon-reduction-plan:2024-03-28:")
    }
    assert len(after) < len(before)
    assert before - after
    assert any(chunk_id.startswith("carbon-reduction-plan:2022-09-15:") for chunk_id in collection.rows)
    assert index.search("15", current_only=False)
    index.close()
