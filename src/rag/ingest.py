"""Ingest parsed policy files into Chroma and the keyword index.

Purpose
-------
Chunks each file, stores every chunk in Chroma, and indexes searchable chunks
for keyword search. Editions of one document stay side by side. The latest
publication date is ``current``.

Contents
--------
- ``ChunkStore``: the collection methods ingest uses.
- ``ingest``: parse, chunk, replace one edition, and set status.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from rag.adapters.chunking import Chunk, Chunker, SectionChunker
from rag.adapters.parsing import DocumentParser, ParsedDocument, PolicyMarkdownParser
from rag.keyword_index import KeywordIndex

Metadata = dict[str, str | bool]


class ChunkStore(Protocol):
    """The Chroma collection methods used to store chunk text and metadata."""

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, str | bool]],
    ) -> None:
        """Insert chunks or replace the text and metadata for those ids.

        Args:
            ids: Chunk ids.
            documents: Chunk text, in the same order.
            metadatas: Metadata for each id.
        """
        ...

    def get(self, *, where: Mapping[str, object]) -> Mapping[str, object]:
        """Return stored chunks that match every metadata field in ``where``.

        Args:
            where: One metadata equality, or ``$and`` of several.

        Returns:
            A mapping with ``ids`` and ``metadatas`` lists.
        """
        ...

    def delete(self, *, ids: Sequence[str]) -> None:
        """Remove chunks.

        Args:
            ids: Chunk ids. Missing ids are ignored.
        """
        ...

    def update(
        self,
        *,
        ids: Sequence[str],
        metadatas: Sequence[Mapping[str, str | bool]],
    ) -> None:
        """Replace the metadata for existing chunks.

        Args:
            ids: Chunk ids.
            metadatas: Full metadata for each id.
        """
        ...


def ingest(
    paths: Sequence[str],
    keyword_index: KeywordIndex,
    collection: ChunkStore,
    parser: DocumentParser | None = None,
    chunker: Chunker | None = None,
) -> list[Chunk]:
    """Parse files, chunk them, and replace the stored edition of each file.

    A new publication date is stored beside older editions. Re-ingesting the
    same document id and version deletes chunk ids that the new chunk set does
    not contain and overwrites ids that remain. Other editions are left in
    place, then their status is set from the latest publication date.

    Args:
        paths: Markdown files to ingest.
        keyword_index: Index for searchable chunk text.
        collection: Store for every chunk, including non-searchable ones.
        parser: File parser. Defaults to ``PolicyMarkdownParser``.
        chunker: Chunking strategy. Defaults to ``SectionChunker``.

    Returns:
        The chunks written for this call, with ids and status set.
    """
    document_parser = parser or PolicyMarkdownParser()
    document_chunker = chunker or SectionChunker()
    parsed = [document_parser.parse(path) for path in paths]
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(parsed):
        by_document[document.document_id].append(index)

    written: list[Chunk] = []
    for document_id, indexes in by_document.items():
        stored = _rows(collection, {"document_id": document_id})
        versions = {str(metadata["version"]) for _, metadata in stored}
        versions.update(parsed[index].version for index in indexes)
        latest = max(versions) if versions else ""
        replaced = {parsed[index].version for index in indexes}
        for index in indexes:
            written.extend(
                _replace_edition(
                    parsed[index],
                    document_chunker,
                    keyword_index,
                    collection,
                    "current" if parsed[index].version == latest else "outdated",
                )
            )
        _refresh_status(collection, keyword_index, stored, latest, replaced)
    return written


def _replace_edition(
    document: ParsedDocument,
    chunker: Chunker,
    keyword_index: KeywordIndex,
    collection: ChunkStore,
    status: str,
) -> list[Chunk]:
    """Replace every stored chunk for one document id and version.

    Args:
        document: Parsed document.
        chunker: Chunking strategy.
        keyword_index: Searchable-text index.
        collection: Full chunk store.
        status: ``current`` or ``outdated`` for this edition.

    Returns:
        Chunks written for this edition.
    """
    chunks = [
        replace(
            chunk,
            chunk_id=f"{document.document_id}:{document.version or 'none'}:{index}",
            status=status,
        )
        for index, chunk in enumerate(chunker.chunk(document))
    ]
    old_ids = [
        chunk_id
        for chunk_id, metadata in _rows(
            collection,
            {"document_id": document.document_id, "version": document.version},
        )
    ]
    new_ids = {chunk.chunk_id for chunk in chunks}
    stale = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
    if old_ids:
        keyword_index.delete(old_ids)
    if stale:
        collection.delete(ids=stale)
    searchable = [chunk for chunk in chunks if chunk.searchable]
    if searchable:
        keyword_index.upsert(
            [chunk.chunk_id for chunk in searchable],
            [chunk.text for chunk in searchable],
            [chunk.status for chunk in searchable],
        )
    if chunks:
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )
    return chunks


def _refresh_status(
    collection: ChunkStore,
    keyword_index: KeywordIndex,
    stored: list[tuple[str, Metadata]],
    latest: str,
    replaced_versions: set[str],
) -> None:
    """Mark stored editions that this call did not replace.

    Args:
        collection: Full chunk store.
        keyword_index: Searchable-text index.
        stored: Rows read before the replacement writes.
        latest: Latest publication date for the document, or empty.
        replaced_versions: Versions written by this call.
    """
    for chunk_id, metadata in stored:
        version = str(metadata["version"])
        if version in replaced_versions:
            continue
        status = "current" if version == latest else "outdated"
        if metadata.get("status") == status:
            continue
        updated = dict(metadata)
        updated["status"] = status
        collection.update(ids=[chunk_id], metadatas=[updated])
        if metadata.get("searchable"):
            keyword_index.set_status([chunk_id], status)


def _rows(collection: ChunkStore, where: Mapping[str, str]) -> list[tuple[str, Metadata]]:
    """Read chunk ids and metadata from the store.

    Args:
        collection: Full chunk store.
        where: Metadata equality filters. Two or more fields are sent as ``$and``.

    Returns:
        Matching rows. Metadata values are strings or bools.
    """
    result = collection.get(where=_chroma_where(where))
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    if not isinstance(ids, list) or not isinstance(metadatas, list):
        return []
    rows: list[tuple[str, Metadata]] = []
    for chunk_id, metadata in zip(ids, metadatas, strict=False):
        if not isinstance(metadata, Mapping):
            continue
        rows.append((str(chunk_id), {str(key): value for key, value in metadata.items()}))
    return rows


def _chroma_where(filters: Mapping[str, str]) -> dict[str, object]:
    """Build a Chroma ``where`` filter.

    Chroma allows one operator at the top level. A single field is that
    equality. Several fields are combined with ``$and``.

    Args:
        filters: Field names and values that must all match.

    Returns:
        A filter Chroma's ``get`` accepts.
    """
    clauses: list[dict[str, str]] = [{key: value} for key, value in filters.items()]
    if len(clauses) == 1:
        return dict(clauses[0])
    return {"$and": clauses}


def _metadata(chunk: Chunk) -> Metadata:
    """Return the Chroma metadata for one chunk.

    Args:
        chunk: Chunk with id and status set.

    Returns:
        Scalar metadata. ``chunk_id`` is stored again beside the collection id.
    """
    return {
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "section_path": chunk.section_path,
        "pages": chunk.pages,
        "version": chunk.version,
        "status": chunk.status,
        "content_type": chunk.content_type,
        "searchable": chunk.searchable,
        "chunk_id": chunk.chunk_id,
    }
