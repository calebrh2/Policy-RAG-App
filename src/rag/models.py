"""Answer returned to the user."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Citation(BaseModel):
    """One retrieved chunk named in the answer."""

    model_config = ConfigDict(extra="forbid")

    document_name: str
    section: str
    source_pages: str


class Answer(BaseModel):
    """Answer text plus the chunks it used."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation]
