"""POST /query returns the generated answer and its sources."""

from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from api.main import app
from rag.generate import Answer, Citation


def test_query_returns_the_answer_and_sources(monkeypatch: MonkeyPatch) -> None:
    def fake_answer(query_text: str) -> Answer:
        assert query_text == "Are plastic cups banned?"
        return Answer(
            supported=True,
            text="Yes. Single-use plastic cups are banned.",
            citations=[
                Citation(
                    chunk_id="cups",
                    document_title="Single-use Plastic-free Policy",
                    section_path="Prohibited items",
                )
            ],
        )

    monkeypatch.setattr("api.main.answer_query", fake_answer)
    client = TestClient(app)

    response = client.post("/query", json={"text": "  Are plastic cups banned?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Yes. Single-use plastic cups are banned."
    assert body["citations"][0]["document_title"] == "Single-use Plastic-free Policy"
    assert body["citations"][0]["section_path"] == "Prohibited items"


def test_blank_query_is_rejected() -> None:
    client = TestClient(app)

    response = client.post("/query", json={"text": "   "})

    assert response.status_code == 422
