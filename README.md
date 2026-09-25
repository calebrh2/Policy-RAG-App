# Policy RAG Lab

Open the repository in the dev container so every contributor develops and tests on the same Linux/Python 3.12 Bookworm environment. Dependencies are managed with `uv`; nobody needs to create or activate a virtual environment manually.

## Intended pipeline

`documents → parsing/chunking → dense + BM25 indexing → reciprocal-rank fusion → cross-encoder reranking → LLM adapter → Pydantic-validated answer with citations`

[Run the pipeline](#run-the-pipeline) executes those stages in order.

## Local environment

Copy `.env.example` to `.env` and point `LLM_BASE_URL` and `LLM_MODEL` at the local chat model. Ollama on the host is the default. The API inside the dev container reaches it at `http://host.docker.internal:11434`. Start that model before generation or the query API. Then either reopen in the dev container or run:

```bash
docker compose run --rm rag-dev
```

## Install dependencies with uv

The dev container runs this automatically. Outside the container, install the standard project and developer tools with:

```bash
uv sync --group dev
```

When working on indexing, embeddings, reranking, or retrieval, install the optional ML stack too:

```bash
uv sync --group dev --group rag
```

The `rag` group installs ChromaDB, Sentence Transformers (and therefore PyTorch), and BM25. Mistral is expected to run locally and is reached through its local HTTP endpoint, so its Python SDK is not installed.

## Run the pipeline

Run these from the repository root after `uv sync --group dev --group rag`. The dev container already has `src` on `PYTHONPATH`.

### Extract the source PDFs

```bash
uv run python scripts/preprocessing.py
```

This reads every PDF in `data/source/RAG-documents` and writes markdown, with page markers and tables, to `data/extracted/RAG-documents`. Those markdown files are already in the repository; run this again only after a source PDF changes. `--input-dir` and `--output-dir` override the two paths.

### Ingest the policy corpus

```bash
uv run python scripts/ingest.py
```

The script reads every markdown file in `data/extracted/RAG-documents`, chunks each file by section, and stores the chunks in two places:

- `data/chromadb` holds the dense embeddings and chunk metadata.
- `data/keyword/chunks.sqlite` holds the BM25 keyword index.

The first run downloads `BAAI/bge-small-en-v1.5` and embeds every chunk. Later runs open those same databases and update them in place.

Re-ingesting a file with the same publication date replaces that edition's chunks. A new publication date for the same policy is stored as another edition beside the older one. The latest date is marked current.

### Score retrieval

After the indexes exist:

```bash
uv run python scripts/evaluate_retrieval.py
```

Each golden question is retrieved from the dense index and the BM25 index, fused with reciprocal rank fusion, and reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`. The chat model is not called. The first run downloads the cross-encoder. The report is written to `runs/retrieval/retrieval-<UTC timestamp>.json`.

### Score generation

With the indexes in place and the local model running:

```bash
uv run python scripts/evaluate_generation.py
```

Unset `LLM_BASE_URL` and `LLM_MODEL` values are filled from `.env`. Each golden question is retrieved, reranked, and answered. Citation ids are checked on the raw completion, then the same model scores faithfulness against the cited chunks and accuracy against the gold answer. The report is written to `runs/generation/generation-<UTC timestamp>.json`.

### Serve the query API

```bash
uv run uvicorn api.main:app --app-dir src --env-file .env --host 127.0.0.1 --port 8000
```

`GET /` is the health check. `POST /query` runs retrieval, reranking, and generation for one question and returns a validated answer with citations. The first request loads the embedding model, the cross-encoder, and the chat client.

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"text": "Are plastic cups banned?"}'
```

### Chat UI

With the API listening on port 8000, start the page from `frontend`:

```bash
npm install
npm run dev
```

Vite serves the UI and proxies `/query` to `http://127.0.0.1:8000`.

## Quality gates

```bash
ruff check .
mypy src
pytest tests/unit tests/integration tests/evaluation
```

Run `uv lock` after changing dependencies and commit the resulting `uv.lock`. The requirements files are retained only as compatibility exports; `pyproject.toml` and `uv.lock` are authoritative.
