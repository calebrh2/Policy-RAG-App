"""Minimal API for the chat UI."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    text: str


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict[str, bool | str]:
    return {"ok": True, "received": body.text}
