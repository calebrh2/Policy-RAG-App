"""Corpus selection reads the flag, then the environment, then the default."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from rag.corpus import CORPORA, markdown_paths, resolve_corpus


def test_default_corpus_is_meridian(monkeypatch: MonkeyPatch) -> None:
    """With no flag and no environment variable, meridian is selected."""
    monkeypatch.delenv("RAG_CORPUS", raising=False)

    corpus = resolve_corpus()

    assert corpus is CORPORA["meridian"]
    assert corpus.documents.name == "RAG-documents"
    assert corpus.chroma.name == "meridian"
    assert corpus.keyword.name == "meridian.sqlite"


def test_environment_selects_coforge_unless_the_flag_is_set(monkeypatch: MonkeyPatch) -> None:
    """``RAG_CORPUS`` applies only when the caller does not pass a name."""
    monkeypatch.setenv("RAG_CORPUS", "coforge")

    assert resolve_corpus() is CORPORA["coforge"]
    assert resolve_corpus("meridian") is CORPORA["meridian"]


def test_unknown_corpus_is_rejected() -> None:
    """A name outside the registry raises."""
    with pytest.raises(ValueError, match="other"):
        resolve_corpus("other")


def test_markdown_paths_lists_only_the_corpus_directory() -> None:
    """Each corpus directory has markdown files, and the lists do not overlap."""
    meridian = {path.name for path in markdown_paths(CORPORA["meridian"])}
    coforge = {path.name for path in markdown_paths(CORPORA["coforge"])}

    assert "POL-HB-600_health_benefits_policy.md" in meridian
    assert "Carbon-Reduction-Plan.md" in coforge
    assert meridian.isdisjoint(coforge)
