"""Tests for data schema, normalization, and benchmark records."""

from __future__ import annotations

from paper_search.data.schema import Document, normalize_text
from paper_search.data.scifact import load_scifact_benchmark


def test_document_full_text() -> None:
    doc = Document(doc_id="1", title="Title", abstract="Abstract body")
    assert doc.full_text == "Title [SEP] Abstract body"


def test_normalize_text() -> None:
    text = "CRISPR-Cas9: A Revolutionary Tool, for DNA-cleavage!"
    tokens = normalize_text(text)
    assert "crispr" in tokens
    assert "cas9" in tokens
    assert "revolutionary" in tokens
    assert "cleavage" in tokens

    # Empty text
    assert normalize_text("") == []


def test_load_scifact_benchmark() -> None:
    docs, queries, qrels = load_scifact_benchmark()
    assert len(docs) == 10
    assert len(queries) == 7
    assert len(qrels) == 7

    doc_ids = {d.doc_id for d in docs}
    query_ids = {q.query_id for q in queries}

    # Verify all qrels point to valid documents and queries
    for q_id, rels in qrels.items():
        assert q_id in query_ids
        for doc_id, rel in rels.items():
            assert doc_id in doc_ids
            assert rel in (0, 1)
