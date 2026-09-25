"""Minimal embed-store-retrieve loop on two known policy passages.

Suggested Approach steps 2-3, before chunking, hybrid search, or reranking.
One sentence from the Carbon Reduction Plan is embedded with
sentence-transformers and stored in a Chroma collection. A second sentence,
from the Single-use Plastic-free Policy, is added. A query for each policy
must return that policy's sentence as the closer hit.

Run from the repository root:

    uv run python scripts/run_minimal_retrieval.py
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE retrieval models expect this prefix on queries only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Passage:
    """One known sentence and the query that should retrieve it."""

    passage_id: str
    source: str
    text: str
    query: str


# Short sentences copied from the extracted policies, not from the chunker.
CARBON = Passage(
    passage_id="carbon-reduction-target",
    source="Carbon-Reduction-Plan.md",
    text=(
        "As an organisation we are committed to assessing our carbon emissions "
        "and to relatively reduce them by 20% by 2030 (from our baseline year) "
        "whilst we grow as a firm."
    ),
    query=(
        "By how much and by which year does the Carbon Reduction Plan commit "
        "to relatively reduce emissions?"
    ),
)
PLASTIC = Passage(
    passage_id="plastic-cups-prohibited",
    source="Single-use-Plastic-free-Policy copy.md",
    text=(
        "The purchase, use, distribution, and sale of plastic plates, cups, "
        "and glasses is prohibited within Coforge premises."
    ),
    query="Are plastic plates, cups, and glasses prohibited within Coforge premises?",
)


def main() -> None:
    """Embed one passage, store it, add the second, and query both."""
    print(f"Loading {MODEL_NAME}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="minimal_passages",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    print("\nStep 2: embed and store one known passage", flush=True)
    _store(collection, model, CARBON)
    _confirm_stored(collection, CARBON)
    _query(collection, model, CARBON.query, CARBON.passage_id)

    print("\nStep 3: add a second passage and retrieve the closer one", flush=True)
    _store(collection, model, PLASTIC)
    print(f"Collection count: {collection.count()}", flush=True)
    _query(collection, model, PLASTIC.query, PLASTIC.passage_id)
    _query(collection, model, CARBON.query, CARBON.passage_id)
    print("\nBoth queries returned the more relevant passage.", flush=True)


def _store(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    passage: Passage,
) -> None:
    """Embed one passage and add it to the collection.

    Args:
        collection: Empty or growing Chroma collection. Callers pass embeddings
            in, so the collection has no embedding function.
        model: SentenceTransformer used for document text. No query prefix.
        passage: Known sentence to store.
    """
    vector = model.encode([passage.text], normalize_embeddings=True)
    collection.add(
        ids=[passage.passage_id],
        documents=[passage.text],
        embeddings=[_floats(vector[0])],
        metadatas=[{"source": passage.source}],
    )
    print(f"Stored {passage.passage_id} from {passage.source}", flush=True)
    print(f"  dimensions: {len(vector[0])}", flush=True)
    print(f"  text: {passage.text}", flush=True)


def _confirm_stored(collection: chromadb.Collection, passage: Passage) -> None:
    """Read the stored passage back by id.

    Args:
        collection: Collection that should contain ``passage``.
        passage: The passage just stored.

    Raises:
        RuntimeError: The id is missing or the stored text differs.
    """
    stored = collection.get(ids=[passage.passage_id], include=["documents"])
    documents = stored.get("documents") or []
    if documents != [passage.text]:
        raise RuntimeError(f"{passage.passage_id} was not stored: {documents}")
    print(f"Confirmed {passage.passage_id} is the only stored passage.", flush=True)
    print(f"Collection count: {collection.count()}", flush=True)


def _query(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    expected_id: str,
) -> None:
    """Embed a query and require the expected passage to be closest.

    Args:
        collection: Passages stored so far.
        model: SentenceTransformer. The BGE query prefix is applied here.
        query: Natural-language question.
        expected_id: Passage id that should rank first.

    Raises:
        RuntimeError: The closest passage is not ``expected_id``.
    """
    vector = model.encode([f"{QUERY_PREFIX}{query}"], normalize_embeddings=True)
    found = collection.query(
        query_embeddings=[_floats(vector[0])],
        n_results=collection.count(),
        include=["documents", "distances", "metadatas"],
    )
    ids = _first(found.get("ids"))
    documents = _first(found.get("documents"))
    distances = _first(found.get("distances"))
    print(f"\nQuery: {query}", flush=True)
    for rank, (passage_id, document, distance) in enumerate(
        zip(ids, documents, distances, strict=True),
        start=1,
    ):
        print(
            f"  {rank}. {passage_id}  cosine_distance={float(distance):.4f}",
            flush=True,
        )
        print(f"     {document}", flush=True)
    if not ids or ids[0] != expected_id:
        raise RuntimeError(f"Expected {expected_id} first, got {ids}")
    print(f"Top hit is {expected_id}.", flush=True)


def _floats(vector: object) -> list[float]:
    """Return one embedding as plain floats.

    Args:
        vector: One encoded row from sentence-transformers.

    Returns:
        Python floats Chroma can store.
    """
    return [float(value) for value in vector]  # type: ignore[union-attr]


def _first(column: object) -> list[object]:
    """Return the first query's result column.

    Args:
        column: A Chroma result field, shaped as a list of lists.

    Returns:
        The inner list for the single query, or an empty list.
    """
    if isinstance(column, list) and column and isinstance(column[0], list):
        return list(column[0])
    return []


if __name__ == "__main__":
    main()
