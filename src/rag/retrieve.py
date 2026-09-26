"""Hybrid retrieval over the dense collection and the keyword index.

Purpose
-------
Runs a dense query and a BM25 query, then fuses the rankings with reciprocal
rank fusion. The returned chunks carry the text and metadata needed for
citations.

Contents
--------
- ``RetrievedChunk``: one fused hit.
- ``ChunkSearch``: the collection methods retrieval calls.
- ``reciprocal_rank_fusion``: combine ranked id lists.
- ``retrieve``: dense search, keyword search, fusion, and hydration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.keyword_index import KeywordIndex

Metadata = dict[str, str | bool]
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by hybrid retrieval, best match first."""

    chunk_id: str
    text: str
    metadata: Metadata
    score: float


class ChunkSearch(Protocol):
    """The collection methods used to search and load chunk text."""

    def count(self) -> int:
        """Return how many chunks are stored.

        Returns:
            The collection size, including chunks the query filter will skip.
        """
        ...

    def query(
        self,
        *,
        query_texts: Sequence[str],
        n_results: int,
        where: Mapping[str, object] | None = None,
        include: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        """Return the nearest stored chunks for each query text.

        Args:
            query_texts: One string per query. Retrieval sends a single query.
            n_results: Maximum hits per query.
            where: Metadata filter. A single equality, or ``$and`` of several.
            include: Result fields to load, such as documents and metadatas.

        Returns:
            A mapping whose ``ids``, ``documents``, and ``metadatas`` values
            are lists of lists, one inner list per query.
        """
        ...

    def get(
        self,
        *,
        ids: Sequence[str],
        include: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        """Return stored chunks by id.

        Args:
            ids: Chunk ids. Missing ids are omitted.
            include: Result fields to load, such as documents and metadatas.

        Returns:
            A mapping whose ``ids``, ``documents``, and ``metadatas`` values
            are flat lists.
        """
        ...


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]]) -> list[tuple[str, float]]:
    """Score chunk ids by their ranks in one or more lists.

    Each appearance of an id adds ``1 / (60 + rank)``. Rank starts at 1.
    Cosine distance and BM25 are ignored. Ties break by chunk id.

    Args:
        rankings: Ranked chunk ids, best first. Lists may overlap.

    Returns:
        ``(chunk_id, score)`` pairs, highest score first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def retrieve(
    query: str,
    collection: ChunkSearch,
    keyword_index: KeywordIndex,
    *,
    limit: int = 10,
    per_list: int = 10,
    current_only: bool = True,
) -> list[RetrievedChunk]:
    """Fuse dense and keyword hits for one query.

    Dense search keeps searchable chunks, and current editions when
    ``current_only`` is true. Keyword search applies the same edition rule.
    The fused list is the candidate pool for a later reranker.

    Args:
        query: User text.
        collection: Dense chunk store.
        keyword_index: BM25 index over searchable chunk text.
        limit: Maximum fused hits to return.
        per_list: Maximum hits to take from each search before fusion.
        current_only: When true, only ``current`` chunks match.

    Returns:
        Fused chunks, best first. A hit with no stored document is omitted.
    """
    if limit < 1:
        return []
    found = {
        chunk_id: (text, metadata)
        for chunk_id, text, metadata in _dense_hits(query, collection, per_list, current_only)
    }
    keyword_ids = [
        hit.chunk_id
        for hit in keyword_index.search(query, limit=per_list, current_only=current_only)
    ]
    fused = reciprocal_rank_fusion([list(found), keyword_ids])[:limit]
    missing = [chunk_id for chunk_id, _score in fused if chunk_id not in found]
    if missing:
        for chunk_id, text, metadata in _rows(
            collection.get(ids=missing, include=["documents", "metadatas"]),
            nested=False,
        ):
            found[chunk_id] = (text, metadata)
    chunks: list[RetrievedChunk] = []
    for chunk_id, score in fused:
        stored = found.get(chunk_id)
        if stored is None:
            continue
        text, metadata = stored
        chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, score=score))
    return chunks


def _dense_hits(
    query: str,
    collection: ChunkSearch,
    per_list: int,
    current_only: bool,
) -> list[tuple[str, str, Metadata]]:
    """Return dense hits in rank order.

    Args:
        query: User text.
        collection: Dense chunk store.
        per_list: Maximum hits to request.
        current_only: When true, only ``current`` chunks match.

    Returns:
        Chunk id, text, and metadata. Empty when the collection is empty or
        ``per_list`` is less than 1.
    """
    if per_list < 1:
        return []
    total = collection.count()
    if total < 1:
        return []
    return _rows(
        collection.query(
            query_texts=[query],
            n_results=min(per_list, total),
            where=_where(current_only),
            include=["documents", "metadatas"],
        ),
        nested=True,
    )


def _where(current_only: bool) -> dict[str, object]:
    """Build the dense-search metadata filter.

    Args:
        current_only: When true, require ``status`` ``current`` as well as
            ``searchable``.

    Returns:
        A single equality, or ``$and`` of ``searchable`` and ``status``.
    """
    if not current_only:
        return {"searchable": True}
    return {"$and": [{"searchable": True}, {"status": "current"}]}


def _rows(result: Mapping[str, object], *, nested: bool) -> list[tuple[str, str, Metadata]]:
    """Read chunk id, text, and metadata from a collection result.

    Args:
        result: A ``query`` result when ``nested`` is true, otherwise a ``get``
            result.
        nested: When true, read the first inner list of each field.

    Returns:
        Rows that have a string id and string text. Missing documents are
        skipped.
    """
    ids = _column(result.get("ids"), nested)
    documents = _column(result.get("documents"), nested)
    metadatas = _column(result.get("metadatas"), nested)
    rows: list[tuple[str, str, Metadata]] = []
    for index, chunk_id in enumerate(ids):
        if not isinstance(chunk_id, str) or index >= len(documents):
            continue
        text = documents[index]
        if not isinstance(text, str):
            continue
        raw = metadatas[index] if index < len(metadatas) else None
        rows.append((chunk_id, text, _metadata(raw)))
    return rows


def _column(value: object, nested: bool) -> list[object]:
    """Return one result column as a flat list.

    Args:
        value: The ``ids``, ``documents``, or ``metadatas`` field.
        nested: When true, unwrap the first query's inner list.

    Returns:
        A flat list, or an empty list when the field has the wrong shape.
    """
    if not isinstance(value, list):
        return []
    if not nested:
        return list(value)
    if not value or not isinstance(value[0], list):
        return []
    return list(value[0])


def _metadata(raw: object) -> Metadata:
    """Copy string and bool metadata fields.

    Args:
        raw: One metadata mapping from the collection, or a missing value.

    Returns:
        Scalar metadata. Non-scalar fields are omitted.
    """
    if not isinstance(raw, Mapping):
        return {}
    metadata: Metadata = {}
    for key, value in raw.items():
        if isinstance(value, (bool, str)):
            metadata[str(key)] = value
    return metadata
