"""Tests for Information Retrieval evaluation metrics and bootstrap comparison."""

from __future__ import annotations

import pytest

from paper_search.data.schema import SearchResult
from paper_search.evaluation.bootstrap import compare_systems_bootstrap
from paper_search.evaluation.metrics import IREvaluator, IRMetrics


def test_evaluate_query_empty() -> None:
    metrics = IREvaluator.evaluate_query([], {})
    assert metrics.ndcg_at_10 == 0.0
    assert metrics.map_at_100 == 0.0
    assert metrics.recall_at_100 == 0.0
    assert metrics.mrr_at_10 == 0.0


def test_evaluate_query_perfect_ranking() -> None:
    results = [
        SearchResult(doc_id="d1", score=1.0, rank=1),
        SearchResult(doc_id="d2", score=0.8, rank=2),
        SearchResult(doc_id="d3", score=0.5, rank=3),
    ]
    qrels = {"d1": 1, "d2": 1}

    metrics = IREvaluator.evaluate_query(results, qrels)

    assert metrics.ndcg_at_10 == pytest.approx(1.0)
    assert metrics.map_at_100 == pytest.approx(1.0)
    assert metrics.recall_at_100 == pytest.approx(1.0)
    assert metrics.recall_at_10 == pytest.approx(1.0)
    assert metrics.precision_at_10 == pytest.approx(0.2)  # 2 relevant out of top 10
    assert metrics.mrr_at_10 == pytest.approx(1.0)


def test_evaluate_query_imperfect_ranking() -> None:
    results = [
        SearchResult(doc_id="d3", score=1.0, rank=1),  # irrelevant
        SearchResult(doc_id="d1", score=0.8, rank=2),  # relevant
        SearchResult(doc_id="d2", score=0.5, rank=3),  # relevant
    ]
    qrels = {"d1": 1, "d2": 1}

    metrics = IREvaluator.evaluate_query(results, qrels)

    assert 0.0 < metrics.ndcg_at_10 < 1.0
    assert metrics.mrr_at_10 == pytest.approx(0.5)  # first relevant at rank 2
    assert metrics.recall_at_100 == pytest.approx(1.0)


def test_evaluate_run() -> None:
    all_qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
    }
    run = {
        "q1": [SearchResult(doc_id="d1", score=1.0, rank=1)],
        "q2": [SearchResult(doc_id="d99", score=1.0, rank=1)],
    }

    mean_metrics = IREvaluator.evaluate_run(run, all_qrels)
    assert isinstance(mean_metrics, IRMetrics)
    assert mean_metrics.ndcg_at_10 == pytest.approx(0.5)
    assert mean_metrics.recall_at_100 == pytest.approx(0.5)
    assert mean_metrics.mrr_at_10 == pytest.approx(0.5)

    empty_metrics = IREvaluator.evaluate_run({}, {})
    assert empty_metrics.ndcg_at_10 == 0.0


def test_compare_systems_bootstrap() -> None:
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }
    baseline_run = {
        "q1": [SearchResult(doc_id="d99", score=0.5, rank=1)],
        "q2": [SearchResult(doc_id="d2", score=0.9, rank=1)],
        "q3": [SearchResult(doc_id="d99", score=0.5, rank=1)],
    }
    candidate_run = {
        "q1": [SearchResult(doc_id="d1", score=0.9, rank=1)],
        "q2": [SearchResult(doc_id="d2", score=0.9, rank=1)],
        "q3": [SearchResult(doc_id="d3", score=0.9, rank=1)],
    }

    comp = compare_systems_bootstrap(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        qrels=qrels,
        n_bootstrap=50,
        random_state=42,
    )

    assert comp.wins == 2
    assert comp.ties == 1
    assert comp.losses == 0
    assert comp.mean_delta > 0.0
    assert comp.ci_lower <= comp.ci_upper
