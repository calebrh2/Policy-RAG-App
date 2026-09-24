from rag.adapters.embeddings import QUERY_INSTRUCTION, SentenceTransformerEmbedder


class _FakeEncoder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str], *, normalize_embeddings: bool = False) -> list[list[float]]:
        assert normalize_embeddings is True
        self.batches.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]


def test_documents_are_embedded_without_the_query_instruction() -> None:
    encoder = _FakeEncoder()
    embedder = SentenceTransformerEmbedder(encoder=encoder)

    vectors = embedder.embed_documents(["Scope 2 is 6,746.70", "Dec 2026 plastic target"])

    assert encoder.batches == [["Scope 2 is 6,746.70", "Dec 2026 plastic target"]]
    assert vectors[0][0] == float(len("Scope 2 is 6,746.70"))
    assert QUERY_INSTRUCTION not in encoder.batches[0][0]


def test_query_uses_the_same_model_with_the_bge_instruction() -> None:
    encoder = _FakeEncoder()
    embedder = SentenceTransformerEmbedder(encoder=encoder)

    vector = embedder.embed_query("What was India Scope 2 in 2023?")

    assert encoder.batches == [[f"{QUERY_INSTRUCTION}What was India Scope 2 in 2023?"]]
    assert len(vector) == 2


def test_empty_document_batch_does_not_call_the_model() -> None:
    encoder = _FakeEncoder()
    embedder = SentenceTransformerEmbedder(encoder=encoder)

    assert embedder.embed_documents([]) == []
    assert encoder.batches == []
