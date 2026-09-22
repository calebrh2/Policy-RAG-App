# Pydantic validation and retry contract

Do not place a structured-output schema in the system prompt. The future LLM adapter must treat provider text as untrusted data and validate it after generation with Pydantic.

## Required answer shape

The future `Answer` schema should include:

- `answer: str`
- `citations: list[Citation]`, where each citation identifies `document`, `section`, and `chunk_id`
- optional `confidence` and `warnings`

## Retry behavior

1. Call the LLM adapter.
2. Parse its returned text into the expected transport format.
3. Validate it with the Pydantic answer schema.
4. Retry only transient provider failures and parse/schema-validation failures, using bounded exponential backoff via Tenacity.
5. On exhaustion, raise a typed generation/validation error that preserves the original failures for logging and tests.

Validation failure must never silently produce an answer. Citation IDs must be checked against the chunks supplied to the generator before returning a validated answer.

