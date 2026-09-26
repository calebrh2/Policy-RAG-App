"""Dense embedding adapter for the policy RAG index.

Purpose
-------
Defines the interface Chroma uses for dense vectors, plus one
SentenceTransformer implementation. The database module depends on this
interface, so the stored model can be swapped without editing Chroma setup.

Contents
--------
- ``DenseEmbedding``: documents and queries are embedded separately.
- ``SentenceTransformerEmbedding``: any SentenceTransformer model name.
- ``default_dense_embedding``: ``BAAI/bge-small-en-v1.5`` with its query prefix.

BM25 keyword search is not created here. The SQLite FTS index ranks chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# BGE retrieval models expect this prefix on queries only.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


class DenseEmbedding(Protocol):
    """A dense embedding model. Document text and query text can differ."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed stored documents.

        Args:
            texts: Document strings to embed.

        Returns:
            One float vector per input string.
        """
        ...

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries.

        Args:
            texts: Query strings to embed.

        Returns:
            One float vector per input string.
        """
        ...


class SentenceTransformerEmbedding:
    """Dense embeddings from a SentenceTransformer model.

    The model name is chosen by the caller. ``query_prompt``, when set, is
    prepended in ``embed_query`` and left off stored documents.
    """

    def __init__(
        self,
        model_name: str,
        query_prompt: str | None = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        """Load a SentenceTransformer model.

        Args:
            model_name: Hugging Face or local SentenceTransformer id.
            query_prompt: Optional text prepended to each query.
            device: Torch device. Defaults to CPU.
            normalize_embeddings: L2-normalize vectors. Defaults to True so
                cosine distance matches the inner product.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "sentence-transformers is not installed. Install the rag group with "
                "`uv sync --group dev --group rag`."
            ) from exc

        self.model_name = model_name
        self.query_prompt = query_prompt
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document text with no query prefix.

        Args:
            texts: Document strings.

        Returns:
            One float vector per document.
        """
        return self._encode(list(texts))

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries, prepending ``query_prompt`` when one was configured.

        Args:
            texts: Query strings.

        Returns:
            One float vector per query.
        """
        if self.query_prompt:
            texts = [f"{self.query_prompt}{text}" for text in texts]
        return self._encode(list(texts))

    def config(self) -> dict[str, str | bool | None]:
        """Return constructor arguments so Chroma can rebuild this model.

        Returns:
            Serializable settings for this instance.
        """
        return {
            "model_name": self.model_name,
            "query_prompt": self.query_prompt,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode strings and return plain Python floats.

        Args:
            texts: Strings to encode.

        Returns:
            One float vector per string.
        """
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [[float(value) for value in vector] for vector in vectors]


def default_dense_embedding() -> SentenceTransformerEmbedding:
    """Return the project's default dense model.

    Uses ``BAAI/bge-small-en-v1.5`` on CPU, with normalized vectors and BGE's
    query instruction applied only at query time.

    Returns:
        A ready ``SentenceTransformerEmbedding``.
    """
    return SentenceTransformerEmbedding(
        model_name=DEFAULT_DENSE_MODEL,
        query_prompt=BGE_QUERY_PROMPT,
        device="cpu",
        normalize_embeddings=True,
    )
