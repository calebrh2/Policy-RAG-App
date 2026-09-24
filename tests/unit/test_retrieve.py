"""Hybrid retrieval fuses dense and keyword ranks and keeps current searchable chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from rag.keyword_index import KeywordIndex
from rag.retrieve import reciprocal_rank_fusion, retrieve


def test_fusion_prefers_a_chunk_in_both_lists_and_breaks_ties_by_id() -> None:
    fused = reciprocal_rank_fusion([["both", "b-only"], ["both", "a-only"]])

    assert [chunk_id for chunk_id, _score in fused] == ["both", "a-only", "b-only"]
    assert fused[1][1] == fused[2][1]


class FakeSearch:
    """In-memory stand-in for the collection methods retrieval calls."""

    def __init__(self, rows: dict[str, dict[str, object]], dense_order: list[str]) -> None:
        self.rows = rows
        self.dense_order = dense_order
        self.loaded_ids: list[str] = []

    def count(self) -> int:
        return len(self.rows)

    def query(
        self,
        *,
        query_texts: Sequence[str],
        n_results: int,
        where: Mapping[str, object] | None = None,
        include: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        del query_texts, include
        filters = _filters(where)
        matched = [
            chunk_id
            for chunk_id in self.dense_order
            if chunk_id in self.rows
            and all(self.rows[chunk_id]["metadata"][key] == value for key, value in filters.items())  # type: ignore[index]
        ][:n_results]
        return {
            "ids": [matched],
            "documents": [[self.rows[chunk_id]["document"] for chunk_id in matched]],
            "metadatas": [[self.rows[chunk_id]["metadata"] for chunk_id in matched]],
        }

    def get(
        self,
        *,
        ids: Sequence[str],
        include: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        del include
        self.loaded_ids.extend(ids)
        present = [chunk_id for chunk_id in ids if chunk_id in self.rows]
        return {
            "ids": present,
            "documents": [self.rows[chunk_id]["document"] for chunk_id in present],
            "metadatas": [self.rows[chunk_id]["metadata"] for chunk_id in present],
        }


def test_retrieve_fuses_lists_and_filters_edition_and_searchable(tmp_path: Path) -> None:
    collection = FakeSearch(
        {
            "both": _row("plastic cups are banned", "current", True),
            "keyword-only": _row("plastic cups in catering", "current", True),
            "dense-only": _row("fleet emissions and net zero", "current", True),
            "outdated": _row("plastic cups from 2022", "outdated", True),
            "cover": _row("Supplier approval boilerplate", "current", False),
        },
        ["both", "outdated", "cover", "dense-only"],
    )
    with KeywordIndex(str(tmp_path / "chunks.sqlite")) as index:
        index.upsert(
            ["both", "keyword-only", "outdated", "ghost"],
            [
                "plastic cups are banned",
                "plastic cups in catering",
                "plastic cups from 2022",
                "plastic cups ghost",
            ],
            ["current", "current", "outdated", "current"],
        )

        hits = retrieve("plastic cups", collection, index)
        by_id = {hit.chunk_id: hit for hit in hits}

        assert "keyword-only" in collection.loaded_ids
        assert by_id["keyword-only"].text == "plastic cups in catering"
        assert by_id["both"].score > by_id["keyword-only"].score
        assert "dense-only" in by_id
        assert "outdated" not in by_id
        assert "cover" not in by_id
        assert "ghost" not in by_id

        including_outdated = {
            hit.chunk_id for hit in retrieve("plastic cups", collection, index, current_only=False)
        }
        assert "outdated" in including_outdated
        assert "cover" not in including_outdated


def _row(text: str, status: str, searchable: bool) -> dict[str, object]:
    return {"document": text, "metadata": {"status": status, "searchable": searchable}}


def _filters(where: Mapping[str, object] | None) -> Mapping[str, object]:
    if where is None:
        return {}
    clauses = where.get("$and")
    if isinstance(clauses, list):
        return {key: value for clause in clauses for key, value in clause.items()}
    return where
