"""Data schemas, records, and text normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Scientific paper corpus record."""

    doc_id: str
    title: str
    abstract: str

    @property
    def full_text(self) -> str:
        """Combine title and abstract with explicit section marker."""
        return f"{self.title} [SEP] {self.abstract}"


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


def normalize_text(text: str) -> list[str]:
    """Apply conservative Unicode normalization and tokenization."""
    norm = unicodedata.normalize("NFKC", text).lower()
    tokens = re.findall(r"[a-z0-9]+", norm)
    return tokens
