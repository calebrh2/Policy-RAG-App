"""Chat adapter for a local OpenAI-compatible model.

Purpose
-------
Defines the interface generation uses to complete a chat, plus one HTTP
implementation. The base URL and model name are chosen by the caller, so this
module does not pick a default model.

Contents
--------
- ``ChatModel``: complete one chat from role and content messages.
- ``OpenAICompatibleChat``: POST ``/v1/chat/completions`` and return the text.
- ``default_chat_model``: build that client from ``LLM_BASE_URL`` and ``LLM_MODEL``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Protocol


class ChatModel(Protocol):
    """A chat model. ``complete`` returns the assistant text."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the assistant text for one chat.

        Args:
            messages: Role and content pairs, in conversation order.

        Returns:
            The assistant message text.
        """
        ...


class OpenAICompatibleChat:
    """Chat completions from an OpenAI-compatible HTTP endpoint.

    The base URL and model name are chosen by the caller. The request asks
    for a JSON object. Ollama on the host is one such endpoint.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 120) -> None:
        """Store the endpoint, model name, and request timeout.

        Args:
            base_url: Endpoint origin, without a path. A trailing slash is
                removed.
            model: Model name the server expects.
            timeout: Seconds to wait for the response. Defaults to 120.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """POST a chat completion and return the assistant text.

        Args:
            messages: Role and content pairs, in conversation order.

        Returns:
            The assistant message text.

        Raises:
            ValueError: The server returned an error, or the body had no
                assistant text.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": message["role"], "content": message["content"]} for message in messages],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Chat completion failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"Chat completion request failed: {exc.reason}") from exc
        return _content(body)


def default_chat_model() -> OpenAICompatibleChat:
    """Return a chat client from the environment.

    Reads ``LLM_BASE_URL`` and ``LLM_MODEL``. Both must be non-empty.

    Returns:
        A ready ``OpenAICompatibleChat``.

    Raises:
        ValueError: Either variable is missing or blank.
    """
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL to the local chat endpoint.")
    return OpenAICompatibleChat(base_url=base_url, model=model)


def _content(body: object) -> str:
    """Read the assistant text from a chat-completion body.

    Args:
        body: Decoded JSON from the server.

    Returns:
        The first choice's message content.

    Raises:
        ValueError: The body is missing choices or content.
    """
    content = _message_content(body)
    if content is None or not content.strip():
        raise ValueError("Chat completion response had no content.")
    return content


def _message_content(body: object) -> str | None:
    """Return the first choice's content, or None when the body is incomplete.

    Args:
        body: Decoded JSON from the server.

    Returns:
        Assistant text, or None when choices or content are missing.
    """
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content
