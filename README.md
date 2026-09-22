# Policy RAG Lab

Implementation-free team scaffold for the RAG lab. Open the repository in the dev container so every contributor develops and tests on the same Linux/Python 3.12 Bookworm environment. Dependencies are managed with `uv`; nobody needs to create or activate a virtual environment manually.

## Intended pipeline

`documents → parsing/chunking → dense + BM25 indexing → reciprocal-rank fusion → cross-encoder reranking → LLM adapter → Pydantic-validated answer with citations`

The source modules and test folders are deliberately empty. Implement features in the sequence described in the lab brief, beginning with the minimal two-document embed/store/retrieve check.

## Local environment

Copy `.env.example` to `.env`, add credentials only when an LLM adapter is implemented, then either reopen in the dev container or run:

```bash
docker compose run --rm rag-dev
```

## Quality gates

```bash
ruff check .
mypy src
pytest tests/unit tests/integration tests/evaluation
```

Run `uv lock` after changing dependencies and commit the resulting `uv.lock`. The requirements files are retained only as compatibility exports; `pyproject.toml` and `uv.lock` are authoritative.

See `docs/pydantic-validation-and-retries.md` before implementing the generation adapter, `docs/adapters.md` for external-service boundaries, and `docs/implementation-plan.md` for module ownership and test boundaries.
