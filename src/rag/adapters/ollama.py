"""Ollama adapter using native JSON-schema structured outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonObject = dict[str, Any]
Transport = Callable[[JsonObject], JsonObject]


class OllamaError(RuntimeError):
    pass


class OllamaAdapter:
    def __init__(
        self,
        model_name: str = "mistral",
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        context_window: int = 8192,
        transport: Transport | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.context_window = context_window
        self._transport = transport or self._http_chat

    def _http_chat(self, payload: JsonObject) -> JsonObject:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result: JsonObject = json.load(response)
                return result
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Start Ollama, pull "
                f"{self.model_name}, and set OLLAMA_BASE_URL to an address reachable "
                "from this process. A dev container normally uses "
                "http://host.docker.internal:11434."
            ) from error

    def generate_structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_schema: Mapping[str, Any],
    ) -> str:
        payload: JsonObject = {
            "model": self.model_name,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "format": dict(response_schema),
            "options": {
                "temperature": 0,
                "num_ctx": self.context_window,
            },
        }
        response = self._transport(payload)
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise OllamaError("Ollama response did not contain message.content") from error
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned empty structured content")
        return content

    def generate(self, prompt: str) -> str:
        response = self._transport(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.context_window},
            }
        )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise OllamaError("Ollama response did not contain message.content") from error
        if not isinstance(content, str):
            raise OllamaError("Ollama response content was not text")
        return content
