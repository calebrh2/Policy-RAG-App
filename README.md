# Policy RAG Lab

A local, version-aware retrieval-augmented generation system for policy documents. It extracts PDFs to structured Markdown, creates section-aware chunks, performs hybrid retrieval, reranks the candidates, and generates grounded answers with inline citations.

## Final pipeline

```text
PDFs
  → PyMuPDF extraction
  → structured Markdown
  → section-first chunks with metadata
  → BGE-small dense embeddings + persistent ChromaDB
  → BM25 keyword retrieval
  → reciprocal-rank fusion
  → cross-encoder reranking
  → top 3 chunks
  → Qwen3 8B (thinking off, one normal generation call)
  → Pydantic validation + citation validation
  → cited answer
```

Ordinary questions use current policy chunks. Historical questions use superseded chunks. Comparison questions use both versions. The router applies deterministic rules first and uses a structured Qwen fallback only for ambiguous version intent.

## Runtime defaults

| Component | Configuration |
|---|---|
| Dense embedding | `BAAI/bge-small-en-v1.5`, 384 dimensions |
| Chunk token limit | 480 BGE tokens |
| Dense store | Persistent ChromaDB |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Final context | Top 3 reranked chunks |
| Generator | `qwen3:8b` through Ollama |
| Thinking | Disabled |
| Claim verifier | Disabled by default to reduce latency |
| Citations | Validated inline markers plus chunk metadata |

The optional second-pass verifier can be enabled with `RAG_VERIFY_GENERATED_CLAIMS=true` for higher-assurance workflows. It is disabled for the final fast configuration because it substantially increases latency.

## Important files

- `scripts/preprocessing.py`: PDF-to-Markdown extraction
- `src/rag/ingestion.py`: structural Markdown chunking
- `src/rag/indexing.py`: validated chunk loading and vector indexing
- `src/rag/retrieval.py`: dense, BM25, RRF, and reranked retrieval
- `src/rag/router.py`: current, historical, and comparison routing
- `src/rag/generation.py`: structured grounded generation and citation validation
- `src/rag/config.py`: central runtime settings
- `data/chunks/chunks.jsonl`: 56 final chunks, 48 searchable
- `data/evaluation/evaluation-set.jsonl`: original development benchmark retained for provenance
- `data/evaluation/evaluation-set-aligned-dev.jsonl`: question-aligned final development set
- `data/evaluation/evaluation-rubrics.json`: source-audited deterministic fact checks
- `artifacts/submission/evaluation/run_qwen3_8b_aligned_dev/`: retained final evaluation output
- `artifacts/submission/screenshots/`: retained submission screenshots
- `artifacts/submission/logs/`: retained terminal evidence

## Setup

```bash
uv sync --group dev --group rag
cp .env.example .env
ollama pull qwen3:8b
```

Ollama must be reachable from the development container at the configured URL. The final configuration sends Ollama `think: false`.

## Commands

```bash
# Extract source PDFs
uv run python scripts/preprocessing.py

# Chunk, validate with the real BGE tokenizer, embed, and persist in ChromaDB
uv run python -m rag.cli ingest

# Inspect retrieved and reranked chunks
uv run python -m rag.cli retrieve \
  "What is the current emissions reduction target?" --top-k 3

# Generate a grounded answer with inline citations
uv run python -m rag.cli ask \
  "What is the current emissions reduction target?"

# Demonstrate the minimal two-record vector loop
uv run python scripts/run_minimal_retrieval.py

# Demonstrate hybrid retrieval outperforming dense-only retrieval
uv run python scripts/compare_retrieval.py

# Demonstrate and correct the planted current-versus-superseded data-quality issue
uv run python scripts/demonstrate_data_quality.py

# Reproduce the final aligned development evaluation
uv run python scripts/run_evaluation.py \
  --dataset data/evaluation/evaluation-set-aligned-dev.jsonl \
  --output-dir artifacts/submission/evaluation/run_qwen3_8b_aligned_dev
```

The evaluation runner keeps the original benchmark as its default. The aligned development set must be selected explicitly so its post-tuning provenance is never hidden.

## Final measured development result

The retained Qwen fast run used 15 fixed, source-audited questions:

| Metric | Result |
|---|---:|
| Questions passed | 15/15 |
| Recall@3 | 100% |
| Key-information answer accuracy | 100% |
| Citation evidence coverage | 100% |
| Mean end-to-end latency | 9.51 seconds per query |
| Pipeline errors | 0 |

This is a development-set result, not an untouched holdout estimate. The question labels are source-audited and still require final human approval before being described as a human-reviewed golden set.

## Chunking strategy

Chunking is section-first and size-second. Markdown headings define semantic boundaries. Complete sections are kept when they fit; large sections split by subsections, numbered groups, paragraphs, and finally sentences. Tables retain titles, headers, units, rows, and footnotes. Searchable chunks target 100–450 tokens and stay below the 480-token BGE application limit. Instead of blind sliding overlap, each chunk repeats its document and section context in `retrieval_text`.

Every chunk keeps document title, version, status, section path, page range, content type, searchability, and a stable chunk ID. Current and superseded Carbon Plan chunks remain separate.

## Data-quality control

The corpus deliberately contains a superseded Carbon Reduction Plan with a conflicting 15%-by-2027 target. Unfiltered retrieval can select that exact lexical match for a current-policy question. Version/status metadata, query routing, and current-status filtering ensure ordinary questions use the active 20%-by-2030 plan. Historical and comparison questions can still retrieve the superseded fixture intentionally.

## Submission evidence

The retained evidence covers:

1. Source documents and planted outdated version
2. Minimal embed/store/retrieve loop
3. Chunking, embedding, and persistent indexing
4. End-to-end grounded answer
5. Dense versus hybrid retrieval
6. Cross-encoder reranking
7. Evaluation set and harness
8. Final Qwen evaluation
9. Data-quality diagnosis and fix
10. Inline citations
11. CI workflow and quality checks

The final evaluation and quality-check screenshots must be retaken because stale baseline screenshots were removed during cleanup.

## Quality checks

```bash
uv run ruff check .
uv run mypy src/rag
uv run pytest -q
```

Real-model integration cases are opt-in and run separately through the evaluation command. `pyproject.toml` and `uv.lock` are authoritative; the requirements files are compatibility exports.
