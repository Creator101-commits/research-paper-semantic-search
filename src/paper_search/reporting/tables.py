"""Markdown table formatters and report generators for IR benchmarks."""

from __future__ import annotations

from paper_search.evaluation.bootstrap import PairedComparisonResult
from paper_search.evaluation.metrics import IRMetrics


def format_ir_benchmark_table(
    metrics_by_system: dict[str, IRMetrics],
    frozen_system: str | None = None,
) -> str:
    """Format information retrieval benchmark results as a Markdown table."""
    lines = [
        "| System | NDCG@10 | MAP@100 | Recall@10 | Recall@100 | MRR@10 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for sys_name, m in metrics_by_system.items():
        label = f"`{sys_name}`" + (" (frozen)" if sys_name == frozen_system else "")
        lines.append(
            f"| {label} | {m.ndcg_at_10:.4f} | {m.map_at_100:.4f} | "
            f"{m.recall_at_10:.4f} | {m.recall_at_100:.4f} | {m.mrr_at_10:.4f} |"
        )
    return "\n".join(lines)


def format_paired_comparison_row(res: PairedComparisonResult, label: str) -> str:
    """Format one paired bootstrap comparison as a Markdown table row."""
    return (
        f"| `{label}` | {res.mean_delta:+.4f} "
        f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}] | "
        f"{res.wins} | {res.ties} | {res.losses} |"
    )


def format_paired_comparison_table(res: PairedComparisonResult, label: str) -> str:
    """Format paired query bootstrap comparison table."""
    lines = [
        "| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |",
        "| :--- | :--- | :--- | :--- | :--- |",
        format_paired_comparison_row(res, label),
    ]
    return "\n".join(lines)


def generate_paper_search_report(
    metrics_by_system: dict[str, IRMetrics],
    paired_comp: PairedComparisonResult,
    dataset_name: str = "fixture",
) -> str:
    """Produce comprehensive Markdown research report."""
    if dataset_name == "scifact":
        corpus_note = (
            "BEIR SciFact corpus; test split uses the SciFact development claims "
            "because official SciFact test labels are not public."
        )
    else:
        corpus_note = (
            "Hand-written synthetic fixture; these are lexical demonstrations, "
            "not official SciFact or neural-model results."
        )
    report = f"""# Research Paper Semantic Search Benchmark Report

{corpus_note}
CPU-safe systems only; neural encoders and cross-encoders run via the Kaggle benchmark.

## Overview and Retrieval Comparison

{format_ir_benchmark_table(metrics_by_system)}

## Statistical Comparison against BM25 Lexical Baseline

{format_paired_comparison_table(paired_comp, "Lexical reranker vs BM25")}
"""
    return report
