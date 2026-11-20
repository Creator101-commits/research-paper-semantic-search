"""Tests for BM25, hashed TF-IDF, neural dense, Hybrid RRF, and reranking."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from paper_search.data.schema import Document
from paper_search.reranking.reranker import LexicalOverlapReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import HashedTfidfRetriever, SentenceTransformerRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever, reciprocal_rank_fusion


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


class BagOfWordsEncoder:
    """Deterministic test encoder implementing the TextEncoder protocol."""

    dim = 16

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        return out


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


def test_hashed_tfidf_retrieval(sample_corpus: list[Document]) -> None:
    dense = HashedTfidfRetriever().index(sample_corpus)
    results = dense.search("transformer language modeling", k=2)

    assert len(results) > 0
    assert results[0].doc_id == "d1"

    # Unindexed error
    unindexed = HashedTfidfRetriever()
    with pytest.raises(RuntimeError, match="not indexed"):
        unindexed.search("test")


def test_sentence_transformer_retriever_with_injected_encoder(
    sample_corpus: list[Document],
) -> None:
    retriever = SentenceTransformerRetriever(
        encoder=BagOfWordsEncoder(), backends=("exact",), name="fake"
    ).index(sample_corpus)

    results = retriever.search("transformer language modeling", k=2)
    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert retriever.embedding_dim == 16
    assert retriever.index_size_bytes > 0
    info = retriever.encoder_info()
    assert info["backends"] == ["exact"]

    # Empty/zero-vector query returns no results
    class ZeroEncoder(BagOfWordsEncoder):
        def _encode(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), self.dim), dtype=np.float32)

    zero_retriever = SentenceTransformerRetriever(
        encoder=ZeroEncoder(), backends=("exact",)
    ).index(sample_corpus)
    assert zero_retriever.search("anything", k=2) == []

    # Invalid backend rejected
    with pytest.raises(ValueError, match="Unknown or empty backends"):
        SentenceTransformerRetriever(encoder=BagOfWordsEncoder(), backends=("bogus",))

    # Unindexed error
    unindexed = SentenceTransformerRetriever(encoder=BagOfWordsEncoder())
    with pytest.raises(RuntimeError, match="not indexed"):
        unindexed.search("test")


def test_faiss_backend_parity(sample_corpus: list[Document]) -> None:
    faiss = pytest.importorskip("faiss")
    del faiss
    retriever = SentenceTransformerRetriever(
        encoder=BagOfWordsEncoder(), backends=("exact", "faiss"), name="fake"
    ).index(sample_corpus)
    for query in ["transformer language modeling", "CRISPR gene editing", "graphs"]:
        exact = retriever.search(query, k=3, backend="exact")
        approx = retriever.search(query, k=3, backend="faiss")
        assert [r.doc_id for r in exact] == [r.doc_id for r in approx]
        assert [r.rank for r in exact] == [r.rank for r in approx]


def test_rrf_fusion_function(sample_corpus: list[Document]) -> None:
    lexical = [
        type("R", (), {"doc_id": "d1", "score": 1.0, "rank": 1})(),
        type("R", (), {"doc_id": "d2", "score": 0.5, "rank": 2})(),
    ]
    dense = [
        type("R", (), {"doc_id": "d2", "score": 1.0, "rank": 1})(),
        type("R", (), {"doc_id": "d3", "score": 0.5, "rank": 2})(),
    ]
    fused = reciprocal_rank_fusion(
        lexical, dense, k_rrf=60.0, lexical_weight=1.0, dense_weight=1.0, k=3
    )
    assert len(fused) == 3
    # d2 appears in both lists at rank 1/2, should rank highly
    assert fused[0].doc_id == "d2"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(lexical, dense, k_rrf=-1.0, k=3)


def test_hybrid_rrf_retrieval(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever(k_rrf=10.0).index(sample_corpus)
    results = hybrid.search("CRISPR genetic modification", k=2)

    assert len(results) > 0
    assert results[0].doc_id == "d2"

    # Hybrid accepts an injected dense backend
    hybrid2 = HybridRRFRetriever(
        dense=HashedTfidfRetriever(embedding_dim=32)
    ).index(sample_corpus)
    assert len(hybrid2.search("CRISPR", k=2)) > 0


def test_lexical_overlap_reranking(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever().index(sample_corpus)
    candidates = hybrid.search("complex networks and graphs", k=3)

    reranker = LexicalOverlapReranker()
    reranker.index_documents(sample_corpus)

    reranked, elapsed_ms = reranker.rerank(
        "complex networks and graphs", candidates, top_k=2, candidate_depth=3
    )

    assert len(reranked) > 0
    assert reranked[0].doc_id == "d3"
    assert elapsed_ms >= 0.0


def test_lexical_rerank_preserves_tail(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever().index(sample_corpus)
    candidates = hybrid.search("neural networks", k=3)
    reranker = LexicalOverlapReranker()
    reranker.index_documents(sample_corpus)

    reranked, _ = reranker.rerank(
        "neural networks", candidates, top_k=3, candidate_depth=1
    )
    # Depth-1 rerank reorders only the head; tail keeps first-stage order.
    assert len(reranked) == 3
    assert [r.rank for r in reranked] == [1, 2, 3]
    assert reranked[1].doc_id == candidates[1].doc_id
    assert reranked[2].doc_id == candidates[2].doc_id


def test_lexical_rerank_scores_nonincreasing(sample_corpus: list[Document]) -> None:
    hybrid = HybridRRFRetriever().index(sample_corpus)
    candidates = hybrid.search("neural networks", k=3)
    reranker = LexicalOverlapReranker()
    reranker.index_documents(sample_corpus)

    reranked, _ = reranker.rerank(
        "neural networks", candidates, top_k=3, candidate_depth=1
    )
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)
