from __future__ import annotations

import json

from rag.adapters.ollama import OllamaAdapter
from rag.models import StructuredAnswerPayload


def test_schema_is_sent_via_format_not_embedded_in_prompt() -> None:
    captured = {}

    def transport(payload):  # type: ignore[no-untyped-def]
        captured.update(payload)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "answer": "Supported answer.",
                        "sufficient_evidence": True,
                        "cited_chunk_ids": ["chunk-1"],
                    }
                )
            }
        }

    adapter = OllamaAdapter(transport=transport)
    content = adapter.generate_structured(
        [{"role": "user", "content": "Use the supplied source."}],
        response_schema=StructuredAnswerPayload.model_json_schema(),
    )

    assert json.loads(content)["answer"] == "Supported answer."
    assert captured["format"] == StructuredAnswerPayload.model_json_schema()
    assert "properties" not in captured["messages"][0]["content"]
    assert captured["options"]["temperature"] == 0
