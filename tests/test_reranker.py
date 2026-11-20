"""Tests for the cross-encoder reranker with an injected pair scorer."""

from __future__ import annotations

import numpy as np

from paper_search.data.schema import Document, SearchResult
from paper_search.reranking.reranker import CrossEncoderReranker


class FakeScorer:
    model_id = "fake-ce"
    revision = "fake-rev"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        # Score = shared token count; deterministic and order-independent.
        q_tokens = set(query.lower().split())
        return np.array(
            [len(q_tokens & set(t.lower().split())) for t in texts],
            dtype=np.float32,
        )


_DOCS = [
    Document(doc_id="d1", title="", abstract="unrelated words"),
    Document(doc_id="d2", title="", abstract="neural networks deep"),
    Document(doc_id="d3", title="", abstract="completely different"),
]


def _candidates() -> list[SearchResult]:
    return [
        SearchResult(doc_id="d1", score=0.9, rank=1),
        SearchResult(doc_id="d2", score=0.8, rank=2),
        SearchResult(doc_id="d3", score=0.7, rank=3),
    ]


def test_cross_encoder_reranker_reorders_head() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer())
    reranker.index_documents(_DOCS)
    reranked, elapsed = reranker.rerank(
        "neural networks", _candidates(), top_k=3, candidate_depth=2
    )
    # d2 shares two tokens with the query and outranks d1.
    assert reranked[0].doc_id == "d2"
    assert elapsed >= 0.0


def test_cross_encoder_reranker_preserves_tail_and_depth() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer())
    reranker.index_documents([])
    candidates = _candidates()
    reranked, _ = reranker.rerank(
        "neural networks", candidates, top_k=3, candidate_depth=1
    )
    # Only the head is rescored; tail keeps first-stage order and ranks.
    assert [r.rank for r in reranked] == [1, 2, 3]
    assert reranked[1].doc_id == candidates[1].doc_id
    assert reranked[2].doc_id == candidates[2].doc_id


def test_cross_encoder_reranker_tie_break_deterministic() -> None:
    class ConstantScorer:
        model_id = "const"
        revision = "const"

        def score(self, query: str, texts: list[str]) -> np.ndarray:
            return np.ones(len(texts), dtype=np.float32)

    reranker = CrossEncoderReranker(scorer=ConstantScorer())
    reranker.index_documents([])
    out1, _ = reranker.rerank("q", _candidates(), top_k=3, candidate_depth=3)
    out2, _ = reranker.rerank("q", _candidates(), top_k=3, candidate_depth=3)
    assert [r.doc_id for r in out1] == [r.doc_id for r in out2]
    # Ties resolve by original candidate order.
    assert [r.doc_id for r in out1] == ["d1", "d2", "d3"]


def test_cross_encoder_reranker_scores_nonincreasing() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer())
    reranker.index_documents(_DOCS)
    reranked, _ = reranker.rerank(
        "neural networks", _candidates(), top_k=3, candidate_depth=1
    )
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_cross_encoder_reranker_score_candidates_alignment() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer())
    reranker.index_documents(_DOCS)
    scores = reranker.score_candidates("neural networks", _candidates()[:2])
    assert scores.shape == (2,)
    # d2 has more overlap than d1.
    assert scores[1] > scores[0]


def test_cross_encoder_reranker_empty_candidates() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer())
    reranker.index_documents(_DOCS)
    reranked, elapsed = reranker.rerank("q", [], top_k=5, candidate_depth=10)
    assert reranked == []
    assert elapsed >= 0.0
