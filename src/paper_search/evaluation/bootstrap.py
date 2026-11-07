"""Paired query bootstrap confidence intervals and win/loss analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper_search.data.schema import SearchResult
from paper_search.evaluation.metrics import IREvaluator


@dataclass(frozen=True)
class PairedComparisonResult:
    """Statistical comparison of two retrieval systems over identical queries."""

    mean_baseline_ndcg: float
    mean_candidate_ndcg: float
    mean_delta: float
    ci_lower: float
    ci_upper: float
    wins: int
    ties: int
    losses: int


def compare_systems_bootstrap(
    baseline_run: dict[str, list[SearchResult]],
    candidate_run: dict[str, list[SearchResult]],
    qrels: dict[str, dict[str, int]],
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> PairedComparisonResult:
    """Compute paired per-query bootstrap difference interval on NDCG@10."""
    if not qrels or n_bootstrap <= 0 or not 0 < alpha < 1:
        raise ValueError("Require judged queries, positive replicates, and alpha in (0, 1).")
    query_ids = sorted(qrels.keys())
    base_scores = np.array(
        [
            IREvaluator.evaluate_query(baseline_run.get(qid, []), qrels[qid]).ndcg_at_10
            for qid in query_ids
        ]
    )
    cand_scores = np.array(
        [
            IREvaluator.evaluate_query(candidate_run.get(qid, []), qrels[qid]).ndcg_at_10
            for qid in query_ids
        ]
    )

    deltas = cand_scores - base_scores
    wins = int(np.sum(deltas > 0.001))
    losses = int(np.sum(deltas < -0.001))
    ties = len(query_ids) - wins - losses

    rng = np.random.default_rng(random_state)
    boot_deltas = np.zeros(n_bootstrap)
    n = len(query_ids)

    for b in range(n_bootstrap):
        sampled_idx = rng.choice(n, size=n, replace=True)
        boot_deltas[b] = float(np.mean(deltas[sampled_idx]))

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    return PairedComparisonResult(
        mean_baseline_ndcg=float(np.mean(base_scores)),
        mean_candidate_ndcg=float(np.mean(cand_scores)),
        mean_delta=float(np.mean(deltas)),
        ci_lower=float(np.percentile(boot_deltas, lower_pct)),
        ci_upper=float(np.percentile(boot_deltas, upper_pct)),
        wins=wins,
        ties=ties,
        losses=losses,
    )
