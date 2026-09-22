# Implementation plan

## Package boundaries

| Area | Future module | Responsibility |
| --- | --- | --- |
| Configuration | `rag/config.py` | Environment-backed settings. |
| Models | `rag/models.py` | All Pydantic models: chunks, citations, requests, answers, and evaluation cases. |
| Adapters | `rag/adapters.py` | Contracts and provider implementations for external systems. |
| Ingestion | `rag/ingestion.py` | Markdown loading, section-aware chunking, metadata, and data-quality lineage. |
| Retrieval | `rag/retrieval.py` | Embedding/indexing, ChromaDB/BM25 search, fusion, and reranking. |
| Generation | `rag/generation.py` | LLM calls, output parsing, schema validation, retries, and citations. |
| Evaluation | `rag/evaluation.py` | Fixed Q&A dataset plus retrieval and answer-quality metrics. |

## Data and test layout

- Put generated Markdown policies in `data/source/`; document the planted quality issue in `docs/data-quality-issue.md`.
- Keep ChromaDB persistence in `data/chromadb/`; it is intentionally ignored by Git.
- Store the fixed 8+ question evaluation dataset in `data/evaluation/`.
- Unit tests mock boundaries and run without models, APIs, or persisted databases.
- Integration tests use a temporary ChromaDB directory and a tiny fixture corpus.
- Evaluation tests run the fixed dataset and report retrieval recall plus answer key-information accuracy.
