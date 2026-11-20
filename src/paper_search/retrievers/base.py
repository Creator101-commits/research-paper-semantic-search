"""Shared retriever interface used by hybrid fusion and the benchmark."""

from __future__ import annotations

from typing import Protocol

from paper_search.data.schema import Document, SearchResult


class Retriever(Protocol):
    """Index a document collection, then return ranked search results."""

    def index(self, documents: list[Document]) -> object: ...

    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...
