"""Reranking: lexical overlap baseline and cross-encoder second stage."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from paper_search.data.schema import Document, SearchResult, normalize_text
from paper_search.reranking.sentence_transformer import SentenceTransformerPairScorer


class LexicalOverlapReranker:
    """Second-stage lexical overlap reranker over top candidate subsets.

    Parameters
    ----------
    title_weight : float, default=2.0
        Importance multiplier for terms matching in the paper title.
    """

    def __init__(self, title_weight: float = 2.0) -> None:
        self.title_weight = title_weight
        self.doc_map: dict[str, Document] = {}
        self._doc_tokens: dict[str, tuple[set[str], set[str]]] = {}

    def index_documents(self, documents: list[Document]) -> None:
        """Register corpus documents and precompute normalized token sets."""
        self.doc_map = {doc.doc_id: doc for doc in documents}
        self._doc_tokens = {
            doc.doc_id: (
                set(normalize_text(doc.title)),
                set(normalize_text(doc.abstract)),
            )
            for doc in documents
        }

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
        candidate_depth: int = 50,
    ) -> tuple[list[SearchResult], float]:
        """Score candidate pairs with title and abstract token overlap."""
        t0 = time.perf_counter()
        query_tokens = set(normalize_text(query))
        if not query_tokens:
            return candidates[:top_k], 0.0

        head = candidates[:candidate_depth]
        tail = candidates[candidate_depth:]
        reranked_scores: list[tuple[int, float, str]] = []

        for pos, cand in enumerate(head):
            title_tokens, abstract_tokens = self._doc_tokens.get(
                cand.doc_id, (set(), set())
            )

            # Fine-grained interaction: title overlap + abstract overlap + first-stage prior
            title_overlap = sum(1 for t in title_tokens if t in query_tokens)
            abstract_overlap = sum(1 for t in abstract_tokens if t in query_tokens)

            # Interaction score
            interaction_score = (
                self.title_weight * title_overlap + abstract_overlap + 0.1 * cand.score
            )
            reranked_scores.append((pos, interaction_score, cand.doc_id))

        # Sort descending by interaction score, deterministic tie-breaking by position
        reranked_scores.sort(key=lambda item: (-item[1], item[0]))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not reranked_scores:
            return tail[:top_k], elapsed_ms

        # Reranked head followed by the untouched tail; ranks are run positions.
        # Tail scores are shifted below the head minimum so score-sorting
        # consumers observe the same order as the rank column.
        head_floor = min(score for _, score, _ in reranked_scores)
        ordered = [(doc_id, score) for _, score, doc_id in reranked_scores]
        ordered.extend(
            (cand.doc_id, head_floor - (r + 1)) for r, cand in enumerate(tail)
        )
        return [
            SearchResult(doc_id=doc_id, score=float(score), rank=r + 1)
            for r, (doc_id, score) in enumerate(ordered[:top_k])
        ], elapsed_ms


class PairScorer(Protocol):
    """Query-document pair scorer returning one score per candidate text."""

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """Cross-encoder reranker over a configurable top-N candidate prefix.

    Candidates beyond ``candidate_depth`` keep their first-stage order, so a
    depth-N rerank reorders only the top N candidates while the run keeps its
    full candidate list for depth-100 metrics.

    Parameters
    ----------
    scorer : PairScorer, optional
        Injected pair scorer; defaults to the pinned MS MARCO cross-encoder.
    """

    def __init__(self, scorer: PairScorer | None = None) -> None:
        self.scorer = scorer if scorer is not None else SentenceTransformerPairScorer()
        self.doc_texts: dict[str, str] = {}

    def index_documents(self, documents: list[Document]) -> None:
        """Register corpus documents for query-passage pair scoring."""
        self.doc_texts = {doc.doc_id: doc.plain_text for doc in documents}

    def score_candidates(
        self, query: str, candidates: list[SearchResult]
    ) -> np.ndarray:
        """Score every candidate; scores align positionally with ``candidates``."""
        texts = [self.doc_texts.get(cand.doc_id, "") for cand in candidates]
        return np.asarray(self.scorer.score(query, texts), dtype=np.float64)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
        candidate_depth: int = 50,
    ) -> tuple[list[SearchResult], float]:
        """Rerank the top ``candidate_depth`` candidates and keep the tail order."""
        t0 = time.perf_counter()
        head = candidates[:candidate_depth]
        tail = candidates[candidate_depth:]
        if not head:
            return tail[:top_k], 0.0

        scores = self.score_candidates(query, head)
        # Sort by cross-encoder score descending; ties keep first-stage order.
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results = [
            SearchResult(
                doc_id=head[i].doc_id,
                score=float(scores[i]),
                rank=r + 1,
            )
            for r, i in enumerate(order)
        ]
        # Tail docs keep first-stage order but are not rescored; their scores
        # are shifted below the reranked head so score-sorting consumers see
        # the same ranking as the rank column. Output ranks are run positions.
        floor = min(r.score for r in results)
        tail_results = [
            SearchResult(
                doc_id=cand.doc_id,
                score=floor - (r + 1),
                rank=len(results) + r + 1,
            )
            for r, cand in enumerate(tail)
        ]
        return (results + tail_results)[:top_k], elapsed_ms
