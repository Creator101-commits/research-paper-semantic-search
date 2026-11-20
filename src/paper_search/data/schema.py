"""Data schemas, records, and text normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Scientific paper corpus record."""

    doc_id: str
    title: str
    abstract: str

    @property
    def full_text(self) -> str:
        """Title and abstract joined with an explicit marker for lexical indexing."""
        return f"{self.title} [SEP] {self.abstract}"

    @property
    def plain_text(self) -> str:
        """Title and abstract joined with a plain space for neural encoders."""
        return " ".join(part for part in (self.title, self.abstract) if part)


@dataclass(frozen=True)
class Query:
    """Benchmark scientific claim or question."""

    query_id: str
    text: str
    split: str = "test"


@dataclass(frozen=True)
class SearchResult:
    """Retrieved document candidate with score and rank."""

    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class BenchmarkDataset:
    """Loaded benchmark corpus, queries, relevance judgments, and provenance."""

    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
    provenance: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> list[str]:
    """Apply conservative Unicode normalization and tokenization."""
    norm = unicodedata.normalize("NFKC", text).lower()
    tokens = re.findall(r"[a-z0-9]+", norm)
    return tokens
