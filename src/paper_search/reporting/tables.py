"""Markdown table formatters and report generators for IR benchmarks."""

from __future__ import annotations

from paper_search.evaluation.bootstrap import PairedComparisonResult
from paper_search.evaluation.metrics import IRMetrics


def format_ir_benchmark_table(metrics_by_system: dict[str, IRMetrics]) -> str:
    """Format information retrieval benchmark results as a Markdown table."""
    lines = [
        "| System | NDCG@10 | MAP@100 | Recall@10 | Recall@100 | MRR@10 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for sys_name, m in metrics_by_system.items():
        lines.append(
            f"| `{sys_name}` | {m.ndcg_at_10:.4f} | {m.map_at_100:.4f} | "
            f"{m.recall_at_10:.4f} | {m.recall_at_100:.4f} | {m.mrr_at_10:.4f} |"
        )
    return "\n".join(lines)


def format_paired_comparison_table(res: PairedComparisonResult, label: str) -> str:
    """Format paired query bootstrap comparison table."""
    lines = [
        "| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |",
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            f"| `{label}` | +{res.mean_delta:.4f} "
            f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}] | "
            f"{res.wins} | {res.ties} | {res.losses} |"
        ),
    ]
    return "\n".join(lines)


def generate_paper_search_report(
    metrics_by_system: dict[str, IRMetrics],
    paired_comp: PairedComparisonResult,
) -> str:
    """Produce comprehensive Markdown research report."""
    report = f"""# Research Paper Semantic Search Benchmark Report

## Overview and Retrieval Comparison

{format_ir_benchmark_table(metrics_by_system)}

## Statistical Comparison against BM25 Lexical Baseline

{format_paired_comparison_table(paired_comp, "Cross-Encoder vs BM25")}
"""
    return report
