# Adapter architecture

Yes—use adapters for every dependency that may change, needs a test fake, or depends on a remote/model-specific API. Keep pipeline orchestration dependent on these interfaces, not on ChromaDB, Sentence Transformers, Mistral, or a specific BM25 package.

## Adapter boundaries

| Adapter | Contract responsibility | Initial implementation | Test replacement |
| --- | --- | --- | --- |
| `embeddings` | Embed document chunks and queries into vectors. | Sentence Transformers | Deterministic in-memory vectors |
| `vector_store` | Upsert chunks, persist metadata, and perform dense top-k search. | ChromaDB | In-memory store / temporary ChromaDB |
| `keyword_search` | Return sparse/BM25 candidates with scores. | `rank-bm25` | Fixed scored fixture |
| `retrieval` | Fuse dense and sparse results (RRF) and deduplicate. | Hybrid retrieval adapter | Deterministic candidate list |
| `reranker` | Score query/chunk candidates and select final context. | Sentence Transformers CrossEncoder | Fixed score fixture |
| `llm` | Generate provider text and normalize provider failures. | Mistral | Scripted fake responses |

The generation service owns Pydantic validation and Tenacity retries. The LLM adapter returns raw provider text only; it must not embed schema instructions in a system prompt or decide whether a response is valid.

Each adapter should receive Pydantic request/response models from `rag.models`, making metadata and citations consistent across implementations. Keep one protocol/abstract contract and its provider implementations together in `rag/adapters.py`; only split that file if it becomes difficult to navigate.
