"""Score golden-set retrieval and write one JSON run.

Run from the repository root, after the indexes exist:

    uv run python scripts/evaluate_retrieval.py

Each golden question is retrieved and reranked once. The ordered chunk ids are
the ones ``generate`` would place in the prompt. The chat model is not called.
The file lands in ``runs/retrieval/retrieval-<UTC timestamp>.json``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.ChromaDB import get_collection
from rag.keyword_index import KeywordIndex
from rag.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)
from rag.rerank import default_reranker, rerank
from rag.retrieve import retrieve
from tests.evaluation.cases import CASES, EvalCase

KEYWORD_PATH = ROOT / "data/keyword/chunks.sqlite"
CHROMA_PATH = ROOT / "data/chromadb"
RUNS = ROOT / "runs" / "retrieval"
PROMPT_K = 5


class ChunkStore(Protocol):
    """The collection method that loads chunk text by metadata."""

    def get(
        self,
        *,
        where: Mapping[str, object] | None = None,
        include: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        """Return stored chunks matching a metadata filter.

        Args:
            where: Metadata filter.
            include: Result fields to load, such as documents and metadatas.

        Returns:
            A mapping whose ``ids``, ``documents``, and ``metadatas`` values
            are flat lists.
        """
        ...


def gold_chunk_id(case: EvalCase, collection: ChunkStore) -> str:
    """Return the one current chunk the case names.

    Args:
        case: A question with a document, section, and needle.
        collection: Dense store that supports ``get`` with a metadata filter.

    Returns:
        The matching chunk id.

    Raises:
        ValueError: The needle matches none or more than one current chunk.
    """
    stored = collection.get(where={"searchable": True}, include=["documents", "metadatas"])
    ids = stored.get("ids")
    documents = stored.get("documents")
    metadatas = stored.get("metadatas")
    matched: list[str] = []
    if isinstance(ids, list) and isinstance(documents, list) and isinstance(metadatas, list):
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            if (
                not isinstance(chunk_id, str)
                or not isinstance(text, str)
                or not isinstance(metadata, dict)
            ):
                continue
            if metadata.get("status") != "current":
                continue
            if metadata.get("document_id") != case.document_id:
                continue
            if metadata.get("section_path") != case.section_path:
                continue
            if case.needle not in text:
                continue
            matched.append(chunk_id)
    if len(matched) != 1:
        raise ValueError(f"{case.case_id} matched {matched}")
    return matched[0]


def main() -> None:
    """Retrieve and rerank each golden question once, then write the run file."""
    total = len(CASES)
    print(f"Scoring {total} golden questions at k={PROMPT_K}", flush=True)
    print("Loading dense collection", flush=True)
    collection = get_collection(persist_directory=str(CHROMA_PATH))
    print("Loading reranker", flush=True)
    reranker = default_reranker()
    queries: list[dict[str, object]] = []
    rankings: list[tuple[list[str], set[str]]] = []
    with KeywordIndex(str(KEYWORD_PATH)) as index:
        for number, case in enumerate(CASES, start=1):
            print(f"[{number}/{total}] {case.case_id}", flush=True)
            relevant = gold_chunk_id(case, collection)
            chunks = rerank(
                case.question,
                retrieve(case.question, collection, index),
                reranker,
                limit=PROMPT_K,
            )
            retrieved = [chunk.chunk_id for chunk in chunks]
            relevant_ids = {relevant}
            rankings.append((retrieved, relevant_ids))
            query_metrics = {
                "recall_at_k": recall_at_k(retrieved, relevant_ids, PROMPT_K),
                "precision_at_k": precision_at_k(retrieved, relevant_ids, PROMPT_K),
                "mrr": reciprocal_rank(retrieved, relevant_ids),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant_ids, PROMPT_K),
            }
            print(
                f"  recall={query_metrics['recall_at_k']:.3f} "
                f"precision={query_metrics['precision_at_k']:.3f} "
                f"mrr={query_metrics['mrr']:.3f} "
                f"ndcg={query_metrics['ndcg_at_k']:.3f}",
                flush=True,
            )
            queries.append(
                {
                    "k": PROMPT_K,
                    "question": case.question,
                    "case_id": case.case_id,
                    "gold_chunk_id": relevant,
                    "retrieved_chunk_ids": retrieved,
                    "metrics": query_metrics,
                }
            )
    scores = score_retrieval(rankings, PROMPT_K)
    print(
        f"Means recall={scores.recall_at_k:.3f} precision={scores.precision_at_k:.3f} "
        f"mrr={scores.mrr:.3f} ndcg={scores.ndcg_at_k:.3f}",
        flush=True,
    )
    payload = {
        "k": PROMPT_K,
        "means": {
            "recall_at_k": scores.recall_at_k,
            "precision_at_k": scores.precision_at_k,
            "mrr": scores.mrr,
            "ndcg_at_k": scores.ndcg_at_k,
        },
        "queries": queries,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"retrieval-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
