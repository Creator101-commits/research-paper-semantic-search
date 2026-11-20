"""Tests for data schema, normalization, and benchmark records."""

from __future__ import annotations

from paper_search.data.fixtures import load_fixture_benchmark
from paper_search.data.schema import Document, normalize_text


def test_document_full_text() -> None:
    doc = Document(doc_id="1", title="Title", abstract="Abstract body")
    assert doc.full_text == "Title [SEP] Abstract body"
    assert doc.plain_text == "Title Abstract body"


def test_document_plain_text_empty_title() -> None:
    doc = Document(doc_id="1", title="", abstract="Abstract body")
    assert doc.plain_text == "Abstract body"


def test_normalize_text() -> None:
    text = "CRISPR-Cas9: A Revolutionary Tool, for DNA-cleavage!"
    tokens = normalize_text(text)
    assert "crispr" in tokens
    assert "cas9" in tokens
    assert "revolutionary" in tokens
    assert "cleavage" in tokens

    # Empty text
    assert normalize_text("") == []


def test_load_fixture_benchmark() -> None:
    data = load_fixture_benchmark()
    docs, queries, qrels = data.documents, data.queries, data.qrels
    assert data.name == "fixture"
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
