"""Information Retrieval metrics: NDCG, MAP, Recall, Precision, MRR."""

from __future__ import annotations

import math
from dataclasses import dataclass

from paper_search.data.schema import SearchResult


@dataclass(frozen=True)
class IRMetrics:
    """Standard evaluation metrics for a single query or mean across queries."""

    ndcg_at_10: float
    map_at_100: float
    recall_at_100: float
    recall_at_10: float
    precision_at_10: float
    mrr_at_10: float


class IREvaluator:
    """Evaluator for benchmark run rankings against ground-truth qrels."""

    @staticmethod
    def evaluate_query(
        results: list[SearchResult],
        qrels: dict[str, int],
        k_ndcg: int = 10,
        k_map: int = 100,
        k_recall: int = 100,
    ) -> IRMetrics:
        """Compute ranking metrics for a single query."""
        if not qrels or not any(rel > 0 for rel in qrels.values()):
            return IRMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_relevant = sum(1 for rel in qrels.values() if rel > 0)

        # 1. NDCG@K
        dcg = 0.0
        for i, res in enumerate(results[:k_ndcg]):
            rel = qrels.get(res.doc_id, 0)
            if rel > 0:
                dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)

        ideal_rels = sorted(qrels.values(), reverse=True)[:k_ndcg]
        idcg = sum(
            (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)
            for i, rel in enumerate(ideal_rels)
            if rel > 0
        )
        ndcg_10 = (dcg / idcg) if idcg > 0 else 0.0

        # 2. MAP@K
        relevant_retrieved = 0
        sum_precisions = 0.0
        for i, res in enumerate(results[:k_map]):
            rel = qrels.get(res.doc_id, 0)
            if rel > 0:
                relevant_retrieved += 1
                sum_precisions += relevant_retrieved / (i + 1)
        map_100 = sum_precisions / total_relevant if total_relevant > 0 else 0.0

        # 3. Recall@K & Precision@10
        retrieved_rel_100 = sum(1 for res in results[:k_recall] if qrels.get(res.doc_id, 0) > 0)
        retrieved_rel_10 = sum(1 for res in results[:10] if qrels.get(res.doc_id, 0) > 0)

        recall_100 = retrieved_rel_100 / total_relevant if total_relevant > 0 else 0.0
        recall_10 = retrieved_rel_10 / total_relevant if total_relevant > 0 else 0.0
        precision_10 = retrieved_rel_10 / 10.0

        # 4. MRR@10
        mrr_10 = 0.0
        for i, res in enumerate(results[:10]):
            if qrels.get(res.doc_id, 0) > 0:
                mrr_10 = 1.0 / (i + 1)
                break

        return IRMetrics(
            ndcg_at_10=float(ndcg_10),
            map_at_100=float(map_100),
            recall_at_100=float(recall_100),
            recall_at_10=float(recall_10),
            precision_at_10=float(precision_10),
            mrr_at_10=float(mrr_10),
        )

    @classmethod
    def evaluate_run(
        cls,
        run_results: dict[str, list[SearchResult]],
        all_qrels: dict[str, dict[str, int]],
    ) -> IRMetrics:
        """Compute macro-averaged metrics across all evaluated queries."""
        query_metrics: list[IRMetrics] = []

        for q_id, q_qrels in all_qrels.items():
            results = run_results.get(q_id, [])
            metrics = cls.evaluate_query(results, q_qrels)
            query_metrics.append(metrics)

        if not query_metrics:
            return IRMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        n = len(query_metrics)
        return IRMetrics(
            ndcg_at_10=sum(m.ndcg_at_10 for m in query_metrics) / n,
            map_at_100=sum(m.map_at_100 for m in query_metrics) / n,
            recall_at_100=sum(m.recall_at_100 for m in query_metrics) / n,
            recall_at_10=sum(m.recall_at_10 for m in query_metrics) / n,
            precision_at_10=sum(m.precision_at_10 for m in query_metrics) / n,
            mrr_at_10=sum(m.mrr_at_10 for m in query_metrics) / n,
        )
