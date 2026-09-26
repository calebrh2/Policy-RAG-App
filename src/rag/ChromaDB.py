# ruff: noqa: N999
"""Persistent Chroma collection for policy-document retrieval.

Purpose
-------
Opens one local Chroma database that stores a dense vector from the dense
embedding adapter. Keyword search lives in the SQLite FTS index. Chunking,
metadata rules, and search are not implemented here.

Contents
--------
- ``get_client``: persistent client at ``CHROMA_PERSIST_DIRECTORY``.
- ``get_collection``: the ``policy_chunks`` collection, created once and reused.
- ``ChromaDenseEmbedding``: adapts ``DenseEmbedding`` to Chroma's interface.

The database files live under ``CHROMA_PERSIST_DIRECTORY`` (default
``data/chromadb``) and are not committed. ``get_or_create_collection`` ignores
a new schema when the collection already exists. Pass the same dense embedding
on later opens. A schema or model change requires deleting ``data/chromadb``
or using a new collection name.
"""

from __future__ import annotations

import os
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions import register_embedding_function

from rag.adapters.embedding import (
    DenseEmbedding,
    SentenceTransformerEmbedding,
    default_dense_embedding,
)

COLLECTION_NAME = "policy_chunks"
DEFAULT_PERSIST_DIRECTORY = "data/chromadb"


@register_embedding_function
class ChromaDenseEmbedding(EmbeddingFunction[Documents]):
    """Chroma dense embedding function backed by a ``DenseEmbedding`` adapter.

    Upserts call ``embed_documents``. Queries call ``embed_query``. This class
    is the only place that speaks Chroma's embedding-function interface.
    """

    def __init__(self, embedding: DenseEmbedding) -> None:
        """Wrap a dense embedding adapter.

        Args:
            embedding: Model that embeds documents and queries.
        """
        self._embedding = embedding

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents for storage.

        Args:
            input: Document strings Chroma is indexing.

        Returns:
            One dense vector per document.
        """
        return _as_chroma_embeddings(self._embedding.embed_documents(list(input)))

    def embed_query(self, input: Documents) -> Embeddings:
        """Embed queries with the adapter's query path.

        Args:
            input: Query strings.

        Returns:
            One dense vector per query.
        """
        return _as_chroma_embeddings(self._embedding.embed_query(list(input)))

    @staticmethod
    def name() -> str:
        """Return the name Chroma uses to reload this function.

        Returns:
            Stable embedding-function name.
        """
        return "dense_embedding_adapter"

    def default_space(self) -> Space:
        """Return cosine distance.

        Normalized dense vectors are compared by angle.

        Returns:
            The distance space name.
        """
        return "cosine"

    def get_config(self) -> dict[str, Any]:
        """Serialize the wrapped model when it exposes ``config``.

        A custom adapter without ``config`` cannot be rebuilt from disk. Pass
        that same instance to ``get_collection`` on later opens.

        Returns:
            Constructor settings, or an empty dict for an opaque adapter.
        """
        config = getattr(self._embedding, "config", None)
        if not callable(config):
            return {}
        raw = config()
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> ChromaDenseEmbedding:
        """Rebuild a SentenceTransformer-backed function from saved settings.

        Args:
            config: Values previously returned by ``get_config``.

        Returns:
            A wrapper around a new ``SentenceTransformerEmbedding``.

        Raises:
            ValueError: The saved collection used an adapter with no config.
        """
        if "model_name" not in config:
            raise ValueError(
                "This collection was created with a custom dense embedding. "
                "Pass that embedding to get_collection()."
            )
        query_prompt = config.get("query_prompt")
        return ChromaDenseEmbedding(
            SentenceTransformerEmbedding(
                model_name=str(config["model_name"]),
                query_prompt=str(query_prompt) if query_prompt is not None else None,
                device=str(config.get("device", "cpu")),
                normalize_embeddings=bool(config.get("normalize_embeddings", True)),
            )
        )

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """Allow the same model settings to be saved again.

        Args:
            old_config: Config already stored on the collection.
            new_config: Config from the embedding passed to this open.
        """
        return


def get_client(persist_directory: str | None = None) -> ClientAPI:
    """Open the local persistent Chroma client.

    Args:
        persist_directory: Directory for database files. Defaults to
            ``CHROMA_PERSIST_DIRECTORY`` or ``data/chromadb``.

    Returns:
        A persistent Chroma client.
    """
    path = persist_directory or os.environ.get("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIRECTORY)
    return chromadb.PersistentClient(path=path)


def get_collection(
    embedding: DenseEmbedding | None = None,
    persist_directory: str | None = None,
) -> Collection:
    """Return the policy collection, creating the dense index if needed.

    The dense index uses ``embedding``, or ``BAAI/bge-small-en-v1.5`` when
    ``embedding`` is omitted. Callers pass document text to the collection;
    Chroma computes the dense embedding. BM25 search uses the keyword index.

    If the collection already exists, Chroma ignores a new schema. Pass the
    same dense embedding used to create it.

    Args:
        embedding: Dense model to use. Defaults to ``default_dense_embedding``.
        persist_directory: Overrides ``CHROMA_PERSIST_DIRECTORY`` for this call.

    Returns:
        The ``policy_chunks`` collection.
    """
    from chromadb import Schema, VectorIndexConfig

    dense = ChromaDenseEmbedding(embedding or default_dense_embedding())
    schema = Schema()
    schema.create_index(
        config=VectorIndexConfig(space="cosine", embedding_function=dense),
    )
    client = get_client(persist_directory)
    # embedding_function=None keeps Chroma from also writing a collection
    # config, which it rejects when a schema is present. The schema does
    # not keep the live function, so the next call attaches it.
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        schema=schema,
        embedding_function=None,
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=cast(Any, dense))


def _as_chroma_embeddings(vectors: list[list[float]]) -> Embeddings:
    """Convert adapter vectors to the numpy arrays Chroma validates.

    Args:
        vectors: Float vectors from a ``DenseEmbedding``.

    Returns:
        One ``float32`` array per vector.
    """
    import numpy as np

    return [np.asarray(vector, dtype=np.float32) for vector in vectors]
