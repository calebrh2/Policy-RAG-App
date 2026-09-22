FROM ghcr.io/astral-sh/uv:0.7.19 AS uv
FROM python:3.12-bookworm

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# Build tooling is required by a few Python packages on Linux.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN uv sync --all-groups --no-install-project

COPY . .

CMD ["uv", "run", "bash"]
