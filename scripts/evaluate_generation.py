"""Score faithfulness, accuracy, citations, and refusal shape.

Run from the repository root, after the indexes exist and the chat model is set:

    uv run python scripts/evaluate_generation.py

Each golden question is retrieved, reranked, and answered once. Citation ids are
checked on the raw completion. The answer text is checked for the expected key
facts. A judge then scores faithfulness against the cited chunks and accuracy
against the gold answer. The file lands in
``runs/generation/generation-<UTC timestamp>.json``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_retrieval import PROMPT_K, gold_chunk_id

from rag.adapters.chat import ChatModel
from rag.adapters.reranker import Reranker
from rag.ChromaDB import get_collection
from rag.generate import Answer, generate
from rag.judge import (
    _ACCURACY_PROMPT,
    _FAITHFULNESS_PROMPT,
    citations_ok,
    contains_key_information,
    judge_accuracy,
    judge_faithfulness,
    raw_citation_ids,
    refusal_ok,
)
from rag.keyword_index import KeywordIndex
from rag.rerank import default_reranker, rerank
from rag.retrieve import retrieve
from tests.evaluation.cases import GENERATION_CASES, EvalCase

KEYWORD_PATH = ROOT / "data/keyword/chunks.sqlite"
CHROMA_PATH = ROOT / "data/chromadb"
RUNS = ROOT / "runs" / "generation"


class _RecordingChat:
    """Forwards one generation call and keeps the assistant text."""

    def __init__(self, inner: ChatModel) -> None:
        """Store the chat model that generation and nothing else will call.

        Args:
            inner: Chat model used for the generation completion.
        """
        self._inner = inner
        self.raw = ""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the assistant text and keep a copy.

        Args:
            messages: Role and content pairs, in conversation order.

        Returns:
            The assistant message text.
        """
        self.raw = self._inner.complete(messages)
        return self.raw


def score_case(
    case: EvalCase,
    collection: object,
    index: KeywordIndex,
    reranker: Reranker,
    model: ChatModel,
) -> dict[str, object]:
    """Answer one golden question and score the reply.

    Args:
        case: A generation question, including refusal cases.
        collection: Dense store passed to retrieval and gold-chunk lookup.
        index: Open keyword index.
        reranker: Cross-encoder used to choose the prompt chunks.
        model: Chat model for generation and both judgments.

    Returns:
        One query record. A generation failure sets every check to false and
        stores the error.
    """
    relevant = gold_chunk_id(case, collection) if case.expect_supported else None
    chunks = rerank(
        case.question,
        retrieve(case.question, collection, index),
        reranker,
        limit=PROMPT_K,
    )
    retrieved = [chunk.chunk_id for chunk in chunks]
    gold_in_prompt = relevant in retrieved if relevant is not None else None
    recorder = _RecordingChat(model)
    try:
        answer = generate(case.question, chunks, recorder)
    except ValueError as exc:
        return _record(
            case,
            gold_chunk_id=relevant,
            gold_in_prompt=gold_in_prompt,
            answer=None,
            raw_ids=[],
            citations=False,
            refusal=False,
            faithful=False,
            faithful_reason="",
            accurate=False,
            accurate_reason="",
            key_information=False,
            error=str(exc),
        )
    try:
        raw_ids = raw_citation_ids(recorder.raw) if recorder.raw else []
        citations = citations_ok(raw_ids, chunks)
    except ValueError:
        raw_ids = []
        citations = False
    faithful = judge_faithfulness(case.question, answer, chunks, model)
    accurate = judge_accuracy(case.question, answer, case.gold_answer, model)
    return _record(
        case,
        gold_chunk_id=relevant,
        gold_in_prompt=gold_in_prompt,
        answer=answer,
        raw_ids=raw_ids,
        citations=citations,
        refusal=refusal_ok(answer),
        faithful=faithful.passed,
        faithful_reason=faithful.reason,
        accurate=accurate.passed,
        accurate_reason=accurate.reason,
        key_information=contains_key_information(
            answer.text,
            case.key_facts,
            exact=not case.expect_supported,
        ),
        error=None,
    )


def main() -> None:
    """Answer each generation question once, then write the run file."""
    total = len(GENERATION_CASES)
    print(f"Scoring {total} generation questions at k={PROMPT_K}", flush=True)
    print("Loading dense collection", flush=True)
    collection = get_collection(persist_directory=str(CHROMA_PATH))
    print("Loading reranker", flush=True)
    reranker = default_reranker()
    print("Loading chat model", flush=True)
    model = _chat_model()
    queries: list[dict[str, object]] = []
    with KeywordIndex(str(KEYWORD_PATH)) as index:
        for number, case in enumerate(GENERATION_CASES, start=1):
            print(f"[{number}/{total}] {case.case_id}", flush=True)
            record = score_case(case, collection, index, reranker, model)
            queries.append(record)
            print(
                f"  citations_ok={record['citations_ok']} refusal_ok={record['refusal_ok']} "
                f"faithful={record['faithful']} accurate={record['accurate']} "
                f"key_information={record['key_information']}",
                flush=True,
            )
    means = _means(queries)
    print(
        "Means "
        f"citations_ok={means['citations_ok']:.3f} refusal_ok={means['refusal_ok']:.3f} "
        f"faithful={means['faithful']:.3f} accurate={means['accurate']:.3f} "
        f"key_information={means['key_information']:.3f}",
        flush=True,
    )
    payload = {
        "faithfulness_prompt": _FAITHFULNESS_PROMPT,
        "accuracy_prompt": _ACCURACY_PROMPT,
        "k": PROMPT_K,
        "means": means,
        "queries": queries,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"generation-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def _chat_model() -> ChatModel:
    """Return the chat model named by the environment.

    Missing variables are filled from the repository ``.env``. Variables
    already set in the process are left unchanged.

    Returns:
        A ready chat client.

    Raises:
        ValueError: ``LLM_BASE_URL`` or ``LLM_MODEL`` is missing.
    """
    _load_dotenv(ROOT / ".env")
    from rag.adapters.chat import default_chat_model

    return default_chat_model()


def _load_dotenv(path: Path) -> None:
    """Set missing environment variables from a dotenv file.

    Args:
        path: A ``.env`` file. A missing file is ignored.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _record(
    case: EvalCase,
    *,
    gold_chunk_id: str | None,
    gold_in_prompt: bool | None,
    answer: Answer | None,
    raw_ids: list[str],
    citations: bool,
    refusal: bool,
    faithful: bool,
    faithful_reason: str,
    accurate: bool,
    accurate_reason: str,
    key_information: bool,
    error: str | None,
) -> dict[str, object]:
    """Build one query record.

    Args:
        case: The generation question.
        gold_chunk_id: The current chunk that states the answer, or None when
            the corpus cannot answer.
        gold_in_prompt: Whether that chunk was among the prompt chunks.
        answer: The validated reply, or None when generation failed.
        raw_ids: Citation ids from the raw completion.
        citations: Whether those ids name sent chunks with document titles.
        refusal: Whether an unsupported answer has the fixed refusal shape.
        faithful: Whether the judge passed faithfulness.
        faithful_reason: The faithfulness failure reason, or an empty string.
        accurate: Whether the judge passed accuracy.
        accurate_reason: The accuracy failure reason, or an empty string.
        key_information: Whether the answer text includes every expected fact.
        error: The generation error, or None when generation returned.

    Returns:
        The JSON object for this question.
    """
    return {
        "case_id": case.case_id,
        "question": case.question,
        "gold_answer": case.gold_answer,
        "expect_supported": case.expect_supported,
        "gold_chunk_id": gold_chunk_id,
        "gold_in_prompt": gold_in_prompt,
        "answer": None if answer is None else answer.model_dump(),
        "raw_citation_ids": raw_ids,
        "citations_ok": citations,
        "refusal_ok": refusal,
        "faithful": faithful,
        "accurate": accurate,
        "key_information": key_information,
        "key_facts": list(case.key_facts),
        "faithfulness_reason": faithful_reason,
        "accuracy_reason": accurate_reason,
        "error": error,
    }


def _means(queries: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Average the boolean checks.

    Args:
        queries: Query records from ``score_case``.

    Returns:
        The mean of each check. All means are zero when ``queries`` is empty.
    """
    return {
        "citations_ok": _rate(queries, "citations_ok"),
        "refusal_ok": _rate(queries, "refusal_ok"),
        "faithful": _rate(queries, "faithful"),
        "accurate": _rate(queries, "accurate"),
        "key_information": _rate(queries, "key_information"),
    }


def _rate(queries: Sequence[Mapping[str, object]], key: str) -> float:
    """Return the fraction of records whose check is true.

    Args:
        queries: Query records.
        key: Boolean field to average.

    Returns:
        The mean. Zero when ``queries`` is empty or a value is not a bool.
    """
    if not queries:
        return 0.0
    total = 0
    for record in queries:
        if record.get(key) is True:
            total += 1
    return total / len(queries)


if __name__ == "__main__":
    main()
