"""Local LLM adapter. The pipeline calls this, not the HTTP endpoint directly."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, cast

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_ATTEMPTS = 3


class LanguageModel(Protocol):
    """Turns a prompt into model text."""

    def complete(self, prompt: str) -> str:
        """Return the model reply."""


class TransientHttpError(Exception):
    """The local model endpoint failed and the call can be tried again."""


class ModelRequestError(Exception):
    """The local model endpoint rejected the call."""


class _Poster(Protocol):
    def __call__(self, url: str, body: bytes) -> bytes:
        """POST JSON and return the response body."""


class OllamaClient:
    """Call a local OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, poster: _Poster | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._poster = poster or _urlopen_post

    @classmethod
    def from_env(cls) -> OllamaClient:
        values = _env_file()
        base_url = os.environ.get("LLM_BASE_URL", "").strip() or values.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "").strip() or values.get("LLM_MODEL", "")
        if not base_url or not model:
            message = "Set LLM_BASE_URL and LLM_MODEL in .env"
            raise ModelRequestError(message)
        return cls(base_url, model)

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self._model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        payload = json.loads(self._post(body))
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            message = "The local model returned an unexpected response"
            raise ModelRequestError(message) from exc
        return str(content)

    @retry(
        retry=retry_if_exception_type((TransientHttpError, TimeoutError, urllib.error.URLError)),
        stop=stop_after_attempt(_ATTEMPTS),
        wait=wait_exponential(multiplier=0.05, max=0.2),
        reraise=True,
    )
    def _post(self, body: bytes) -> bytes:
        return self._poster(f"{self._base_url}/chat/completions", body)


def _env_file() -> dict[str, str]:
    path = Path(".env")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _urlopen_post(url: str, body: bytes) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return cast(bytes, response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 429 or exc.code >= 500:
            raise TransientHttpError(f"HTTP {exc.code}") from exc
        raise ModelRequestError(f"HTTP {exc.code}") from exc
