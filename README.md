# Policy RAG Lab

Implementation-free team scaffold for the RAG lab. Open the repository in the dev container so every contributor develops and tests on the same Linux/Python 3.12 Bookworm environment. Dependencies are managed with `uv`; nobody needs to create or activate a virtual environment manually.

## Intended pipeline

`documents → parsing/chunking → dense + BM25 indexing → reciprocal-rank fusion → cross-encoder reranking → LLM adapter → Pydantic-validated answer with citations`

The source modules and test folders are deliberately empty. Implement features in the sequence described in the lab brief, beginning with the minimal two-document embed/store/retrieve check.

## Local environment

Copy `.env.example` to `.env`, set the local Mistral endpoint when its adapter is implemented, then either reopen in the dev container or run:

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

## Ingest the policy corpus

From the repository root, after the `rag` group is installed:

```bash
uv run python scripts/ingest.py
```

The script reads every markdown file in `data/extracted/RAG-documents`, chunks each file by section, and stores the chunks in two places:

- `data/chromadb` holds the dense embeddings and chunk metadata.
- `data/keyword/chunks.sqlite` holds the BM25 keyword index.

The first run downloads `BAAI/bge-small-en-v1.5` and embeds every chunk. Later runs open those same databases and update them in place.

Re-ingesting a file with the same publication date replaces that edition's chunks. A new publication date for the same policy is stored as another edition beside the older one. The latest date is marked current.

## Quality gates

```bash
ruff check .
mypy src
pytest tests/unit tests/integration tests/evaluation
```

Run `uv lock` after changing dependencies and commit the resulting `uv.lock`. The requirements files are retained only as compatibility exports; `pyproject.toml` and `uv.lock` are authoritative.
