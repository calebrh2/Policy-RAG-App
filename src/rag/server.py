"""HTTP endpoint that returns the same JSON as the CLI."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

from rag.adapters.ollama import LanguageModel
from rag.adapters.reranker import Reranker
from rag.cli import _Retriever, respond


class _Handler(Protocol):
    def __call__(self, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        """Return a status code and a JSON object."""


def ask_payload(
    body: dict[str, object],
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
) -> tuple[int, dict[str, object]]:
    """Answer one question, or reject an empty body."""
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return 400, {"error": "Question is required.", "answer": "", "citations": []}
    return 200, respond(question.strip(), retriever, reranker, model)


def serve(
    retriever: _Retriever,
    reranker: Reranker,
    model: LanguageModel,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Serve POST /ask until the process stops."""

    def handle(body: dict[str, object]) -> tuple[int, dict[str, object]]:
        return ask_payload(body, retriever, reranker, model)

    server = ThreadingHTTPServer((host, port), _request_handler(handle))
    print(f"Policy chat listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def _request_handler(handle: _Handler) -> type[BaseHTTPRequestHandler]:
    class AskHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/ask":
                self._json(404, {"error": "Not found.", "answer": "", "citations": []})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                self._json(400, {"error": "Question is required.", "answer": "", "citations": []})
                return
            if not isinstance(parsed, dict):
                self._json(400, {"error": "Question is required.", "answer": "", "citations": []})
                return
            status, payload = handle(parsed)
            self._json(status, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    return AskHandler


def main() -> None:
    from rag.cli import _stack

    retriever, reranker, model = _stack()
    serve(retriever, reranker, model)


if __name__ == "__main__":
    main()
