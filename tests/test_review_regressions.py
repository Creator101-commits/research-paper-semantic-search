"""Regression tests for issues found during the correctness audit."""

from __future__ import annotations

from paper_search.data.fixtures import load_fixture_benchmark
from paper_search.evaluation.bootstrap import compare_systems_bootstrap
from paper_search.evaluation.metrics import IREvaluator
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import HashedTfidfRetriever


def test_bm25_does_not_crash_on_empty_tokens() -> None:
    """Regression: BM25 must not divide by zero or crash on punctuation-only queries."""
    docs, queries, qrels = (
        load_fixture_benchmark().documents,
        load_fixture_benchmark().queries,
        load_fixture_benchmark().qrels,
    )
    bm25 = BM25Retriever().index(docs)

    results = bm25.search("!!! ??? ...", k=5)
    assert isinstance(results, list)

    runs = {q.query_id: bm25.search(q.text, k=100) for q in queries}
    metrics = IREvaluator.evaluate_run(runs, qrels)
    assert metrics.ndcg_at_10 >= 0.0
    assert metrics.recall_at_100 >= 0.0


def test_qrels_coverage_is_complete() -> None:
    """Regression: every query in the fixture must have qrels."""
    data = load_fixture_benchmark()
    query_ids = {q.query_id for q in data.queries}
    assert set(data.qrels.keys()) == query_ids


def test_bootstrap_is_deterministic() -> None:
    """Regression: bootstrap confidence intervals must be reproducible."""
    docs = load_fixture_benchmark().documents
    queries = [q for q in load_fixture_benchmark().queries if q.split == "test"]
    qrels = {q.query_id: load_fixture_benchmark().qrels[q.query_id] for q in queries}

    bm25 = BM25Retriever().index(docs)
    dense = HashedTfidfRetriever().index(docs)

    run_a = {q.query_id: bm25.search(q.text, k=100) for q in queries}
    run_b = {q.query_id: dense.search(q.text, k=100) for q in queries}

    res1 = compare_systems_bootstrap(run_a, run_b, qrels, n_bootstrap=100, random_state=42)
    res2 = compare_systems_bootstrap(run_a, run_b, qrels, n_bootstrap=100, random_state=42)

    assert res1.mean_delta == res2.mean_delta
    assert res1.ci_lower == res2.ci_lower
    assert res1.ci_upper == res2.ci_upper
    assert res1.wins == res2.wins
    assert res1.losses == res2.losses
