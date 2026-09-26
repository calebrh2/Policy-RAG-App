"""On-disk BM25 keyword index for policy chunks.

Purpose
-------
Chroma's local engine cannot store a sparse vector index. This module keeps
a SQLite FTS5 table beside that database and ranks the same chunk ids with
FTS5's ``bm25()``.

Contents
--------
- ``KeywordHit``: one ranked chunk id.
- ``KeywordIndex``: create, replace, delete, and search chunks. Search keeps
  ``current`` chunks unless the caller asks for older editions.

The database file defaults to ``data/keyword/chunks.sqlite`` (override with
``KEYWORD_INDEX_PATH``). It is not committed. The table is created on first
open and reused after that.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

DEFAULT_INDEX_PATH = "data/keyword/chunks.sqlite"
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class KeywordHit:
    """A chunk returned by keyword search, best match first."""

    chunk_id: str
    score: float


class KeywordIndex:
    """Persistent FTS5 index keyed by the same ids stored in Chroma."""

    def __init__(self, path: str | None = None) -> None:
        """Open or create the keyword database.

        Args:
            path: SQLite file path. Defaults to ``KEYWORD_INDEX_PATH`` or
                ``data/keyword/chunks.sqlite``.
        """
        self.path = path or os.environ.get("KEYWORD_INDEX_PATH", DEFAULT_INDEX_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                status UNINDEXED,
                tokenize = 'porter'
            )
            """
        )
        columns = [row[1] for row in self._connection.execute("PRAGMA table_info(chunks_fts)")]
        if "status" not in columns:
            raise ValueError(
                f"Keyword index at {self.path} is missing the status column. "
                "Delete that file so it can be recreated."
            )

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        """Return this index for use in a ``with`` block."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the connection when the ``with`` block ends."""
        self.close()

    def upsert(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        statuses: Sequence[str] | None = None,
    ) -> None:
        """Insert chunks or replace the text already stored for those ids.

        Args:
            chunk_ids: Ids shared with the Chroma collection.
            texts: Document text for each id, in the same order.
            statuses: ``current`` or ``outdated`` for each id. Defaults to
                ``current``.

        Raises:
            ValueError: Lengths differ, or an id is empty.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        if statuses is None:
            statuses = ["current"] * len(chunk_ids)
        if len(statuses) != len(chunk_ids):
            raise ValueError("chunk_ids and statuses must have the same length")
        with self._connection:
            for chunk_id, text, status in zip(chunk_ids, texts, statuses, strict=True):
                if not chunk_id:
                    raise ValueError("chunk_id must be non-empty")
                self._connection.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk_id,),
                )
                self._connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, text, status) VALUES (?, ?, ?)",
                    (chunk_id, text, status),
                )

    def set_status(self, chunk_ids: Sequence[str], status: str) -> None:
        """Set the edition status on chunks that are already indexed.

        Args:
            chunk_ids: Ids to update. Missing ids are ignored.
            status: ``current`` or ``outdated``.
        """
        with self._connection:
            self._connection.executemany(
                "UPDATE chunks_fts SET status = ? WHERE chunk_id = ?",
                [(status, chunk_id) for chunk_id in chunk_ids],
            )

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks from the index.

        Args:
            chunk_ids: Ids to delete. Missing ids are ignored.
        """
        with self._connection:
            self._connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

    def search(self, query: str, limit: int = 10, current_only: bool = True) -> list[KeywordHit]:
        """Rank chunks by BM25 against a plain-text query.

        Query words are stemmed with the same Porter tokenizer as the index.
        A chunk matches when it contains any of those words. Higher ``score``
        is a stronger match. Outdated editions are skipped unless
        ``current_only`` is false.

        Args:
            query: User text. Punctuation is ignored.
            limit: Maximum hits to return.
            current_only: When true, only ``current`` chunks match.

        Returns:
            Hits ordered best first. Empty when the query has no words or
            ``limit`` is less than 1.
        """
        match = _match_query(query)
        if not match or limit < 1:
            return []
        status_clause = "AND status = 'current'" if current_only else ""
        rows = self._connection.execute(
            f"""
            SELECT chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            {status_clause}
            ORDER BY score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [KeywordHit(chunk_id=str(chunk_id), score=-float(score)) for chunk_id, score in rows]


def _match_query(query: str) -> str:
    """Turn plain text into an FTS5 OR query.

    Args:
        query: Raw search text.

    Returns:
        A ``MATCH`` expression, or an empty string when there are no words.
    """
    tokens = [token.replace('"', '""') for token in _TOKEN.findall(query)]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)
