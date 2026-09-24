"""Keyword index stores chunks on disk and ranks them with FTS5 BM25."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag.keyword_index import KeywordIndex


def test_search_ranks_the_chunk_that_shares_more_query_terms(tmp_path: Path) -> None:
    index = KeywordIndex(str(tmp_path / "chunks.sqlite"))
    index.upsert(
        ["carbon", "water", "plastic"],
        [
            "Carbon reduction plan for fleet emissions and net zero.",
            "Water management covers leaks, metering, and drought response.",
            "Single-use plastic is banned in catering, including cutlery and cups.",
        ],
    )

    hits = index.search("plastic cups")

    assert [hit.chunk_id for hit in hits] == ["plastic"]
    assert hits[0].score > 0
    index.close()


def test_index_survives_reopen_and_replaces_text(tmp_path: Path) -> None:
    path = str(tmp_path / "chunks.sqlite")
    with KeywordIndex(path) as index:
        index.upsert(["carbon"], ["Carbon reduction plan for fleet emissions."])

    with KeywordIndex(path) as index:
        assert index.search("emissions")[0].chunk_id == "carbon"
        index.upsert(["carbon"], ["Only water stewardship is discussed here."])
        assert index.search("emissions") == []
        assert index.search("water")[0].chunk_id == "carbon"
        index.delete(["carbon"])
        assert index.search("water") == []


def test_porter_stem_matches_plural_query(tmp_path: Path) -> None:
    with KeywordIndex(str(tmp_path / "chunks.sqlite")) as index:
        index.upsert(["plastic"], ["Single-use plastic is banned."])
        assert index.search("plastics")[0].chunk_id == "plastic"


def test_punctuation_query_does_not_error(tmp_path: Path) -> None:
    with KeywordIndex(str(tmp_path / "chunks.sqlite")) as index:
        index.upsert(["carbon"], ["Carbon reduction plan."])
        assert index.search("carbon-reduction!")[0].chunk_id == "carbon"
        assert index.search("???") == []
        assert index.search("carbon", limit=0) == []


def test_old_schema_without_status_asks_for_a_new_file(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize = 'porter')"
    )
    connection.close()

    with pytest.raises(ValueError, match="status column"):
        KeywordIndex(str(path))


def test_upsert_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with (
        KeywordIndex(str(tmp_path / "chunks.sqlite")) as index,
        pytest.raises(ValueError, match="same length"),
    ):
        index.upsert(["a"], ["one", "two"])
