"""Tests for BM25, Dense, Hybrid RRF, and Cross-Encoder reranking."""

from __future__ import annotations

import pytest

from paper_search.data.schema import Document
from paper_search.reranking.reranker import CrossEncoderReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever


@pytest.fixture
def sample_corpus() -> list[Document]:
    return [
        Document(
            doc_id="d1",
            title="Deep Neural Networks for Natural Language Processing",
            abstract=(
                "Transformer architectures revolutionize machine translation "
                "and language modeling."
            ),
        ),
        Document(
            doc_id="d2",
            title="CRISPR gene editing in human therapeutics",
            abstract=(
                "Targeted genetic modification with Cas9 nucleases enables curative "
                "interventions."
            ),
        ),
        Document(
            doc_id="d3",
            title="Statistical Mechanics of Complex Networks",
            abstract="Phase transitions and percolation theory on scale-free graphs.",
        ),
    ]


def test_bm25_retrieval(sample_corpus: list[Document]) -> None:
    bm25 = BM25Retriever().index(sample_corpus)

    # Search for NLP
    results = bm25.search("transformer language modeling", k=2)
    assert len(results) > 0
    assert results[0].doc_id == "d1"

    # Search for CRISPR
    crispr_results = bm25.search("CRISPR gene editing", k=1)
    assert len(crispr_results) == 1
    assert crispr_results[0].doc_id == "d2"

    # Unindexed error
    unindexed = BM25Retriever()
    with pytest.raises(RuntimeError, match="not indexed"):
        unindexed.search("test")


def test_dense_retrieval(sample_corpus: list[Document]) -> None:
    dense = DenseRetriever().index(sample_corpus)
    results = dense.search("transformer language modeling", k=2)

    assert len(results) > 0
    assert results[0].doc_id == "d1"

    # Unindexed error
    unindexed = DenseRetriever()
    with pytest.raises(RuntimeError, match="not indexed"):
        unindexed.search("test")


def test_hybrid_rrf_retrieval(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever(k_rrf=10.0).index(sample_corpus)
    results = hybrid.search("CRISPR genetic modification", k=2)

    assert len(results) > 0
    assert results[0].doc_id == "d2"


def test_cross_encoder_reranking(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever().index(sample_corpus)
    candidates = hybrid.search("complex networks and graphs", k=3)

    reranker = CrossEncoderReranker()
    reranker.index_documents(sample_corpus)

    reranked, elapsed_ms = reranker.rerank(
        "complex networks and graphs", candidates, top_k=2, max_candidate_depth=3
    )

    assert len(reranked) > 0
    assert reranked[0].doc_id == "d3"
    assert elapsed_ms >= 0.0
