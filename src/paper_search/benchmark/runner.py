"""End-to-end SciFact benchmark: dev-only tuning, frozen test evaluation.

All model selection (dense encoder for hybrid, RRF constant and weights,
rerank depth) is tuned on the deterministic development split carved from
SciFact train queries. Test qrels are touched exactly once, with the frozen
configuration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import paper_search
from paper_search.benchmark.models import (
    DENSE_MODEL_SPECS,
    DenseModelSpec,
    dense_model_to_dict,
    reranker_model_to_dict,
)
from paper_search.data.schema import BenchmarkDataset, Query, SearchResult
from paper_search.data.scifact import load_scifact_benchmark
from paper_search.data.splits import make_dev_split
from paper_search.evaluation.bootstrap import (
    PairedComparisonResult,
    compare_systems_bootstrap,
)
from paper_search.evaluation.metrics import IREvaluator, IRMetrics
from paper_search.reporting.tables import (
    format_ir_benchmark_table,
    format_paired_comparison_row,
)
from paper_search.reranking.reranker import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    PairScorer,
)
from paper_search.reranking.sentence_transformer import SentenceTransformerPairScorer
from paper_search.retrievers.base import Retriever
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import (
    HashedTfidfRetriever,
    SentenceTransformerEncoder,
    SentenceTransformerRetriever,
    TextEncoder,
)
from paper_search.retrievers.hybrid import reciprocal_rank_fusion

EncoderFactory = Callable[[DenseModelSpec], TextEncoder]
ScorerFactory = Callable[[], PairScorer]


def _default_encoder_factory(spec: DenseModelSpec) -> TextEncoder:
    return SentenceTransformerEncoder(
        model_id=spec.model_id,
        revision=spec.revision,
        query_prompt=spec.query_prompt,
    )


def _default_scorer_factory() -> PairScorer:
    return SentenceTransformerPairScorer()


@dataclass(frozen=True)
class LatencyStats:
    """Per-query latency summary in milliseconds (warmup query excluded)."""

    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    total_ms: float

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> LatencyStats:
        if not samples_ms:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(samples_ms, dtype=np.float64)
        return cls(
            count=int(arr.size),
            mean_ms=float(arr.mean()),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            max_ms=float(np.max(arr)),
            total_ms=float(arr.sum()),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class EvaluatedSystem:
    """One named system evaluated on one split."""

    name: str
    split: str
    runs: dict[str, list[SearchResult]]
    metrics: IRMetrics
    latency: LatencyStats | None
    config: dict[str, Any] = field(default_factory=dict)
    per_query_ms: dict[str, float] = field(default_factory=dict)
    # Rerank-stage-only latency for reranked systems; ``latency`` stays
    # end-to-end (first-stage retrieval plus rerank) for every system.
    stage_latency: LatencyStats | None = None


def _measure_searches(
    search_fn: Callable[[str], list[SearchResult]],
    queries: Sequence[Query],
) -> tuple[dict[str, list[SearchResult]], dict[str, float], LatencyStats]:
    """Run ``search_fn`` per query; first query is an untracked warmup."""
    runs: dict[str, list[SearchResult]] = {}
    per_query_ms: dict[str, float] = {}
    samples: list[float] = []
    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        results = search_fn(query.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        runs[query.query_id] = results
        if i > 0:
            per_query_ms[query.query_id] = elapsed_ms
            samples.append(elapsed_ms)
    return runs, per_query_ms, LatencyStats.from_samples(samples)


def _evaluate_retriever(
    name: str,
    split: str,
    retriever: Retriever,
    queries: Sequence[Query],
    qrels: dict[str, dict[str, int]],
    k: int,
    config: dict[str, Any] | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> EvaluatedSystem:
    kwargs = search_kwargs or {}

    def search_fn(text: str) -> list[SearchResult]:
        return retriever.search(text, k=k, **kwargs)

    runs, per_query_ms, latency = _measure_searches(search_fn, queries)
    metrics = IREvaluator.evaluate_run(runs, qrels)
    return EvaluatedSystem(
        name=name,
        split=split,
        runs=runs,
        metrics=metrics,
        latency=latency,
        config=config or {},
        per_query_ms=per_query_ms,
    )


def _evaluate_reranker(
    name: str,
    split: str,
    reranker: LexicalOverlapReranker | CrossEncoderReranker,
    candidates_by_query: dict[str, list[SearchResult]],
    queries: Sequence[Query],
    qrels: dict[str, dict[str, int]],
    k: int,
    candidate_depth: int,
    first_stage_ms: dict[str, float] | None = None,
) -> EvaluatedSystem:
    """Evaluate a reranker over precomputed first-stage candidate lists.

    ``latency`` covers the end-to-end pipeline: the shared first-stage
    per-query times in ``first_stage_ms`` plus the rerank step. The
    rerank step alone is kept in ``stage_latency``.
    """
    first_stage = first_stage_ms or {}
    runs: dict[str, list[SearchResult]] = {}
    per_query_ms: dict[str, float] = {}
    samples: list[float] = []
    stage_samples: list[float] = []
    for i, query in enumerate(queries):
        candidates = candidates_by_query[query.query_id]
        t0 = time.perf_counter()
        results, _ = reranker.rerank(
            query.text, candidates, top_k=k, candidate_depth=candidate_depth
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        runs[query.query_id] = results
        if i > 0:
            e2e_ms = elapsed_ms + first_stage.get(query.query_id, 0.0)
            per_query_ms[query.query_id] = e2e_ms
            samples.append(e2e_ms)
            stage_samples.append(elapsed_ms)
    metrics = IREvaluator.evaluate_run(runs, qrels)
    return EvaluatedSystem(
        name=name,
        split=split,
        runs=runs,
        metrics=metrics,
        latency=LatencyStats.from_samples(samples),
        stage_latency=LatencyStats.from_samples(stage_samples),
        config={"candidate_depth": candidate_depth, "first_stage_k": k},
        per_query_ms=per_query_ms,
    )


def _fuse_runs(
    lexical_runs: dict[str, list[SearchResult]],
    dense_runs: dict[str, list[SearchResult]],
    queries: Sequence[Query],
    *,
    k_rrf: float,
    lexical_weight: float,
    dense_weight: float,
    k: int,
) -> dict[str, list[SearchResult]]:
    """Fuse cached first-stage runs; used for grid search and the frozen hybrid."""
    fused: dict[str, list[SearchResult]] = {}
    for query in queries:
        fused[query.query_id] = reciprocal_rank_fusion(
            lexical_runs[query.query_id],
            dense_runs[query.query_id],
            k_rrf=k_rrf,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            k=k,
        )
    return fused


def _metrics_record(metrics: IRMetrics) -> dict[str, float]:
    return {
        "ndcg_at_10": metrics.ndcg_at_10,
        "map_at_100": metrics.map_at_100,
        "recall_at_100": metrics.recall_at_100,
        "recall_at_10": metrics.recall_at_10,
        "precision_at_10": metrics.precision_at_10,
        "mrr_at_10": metrics.mrr_at_10,
    }


def _per_query_metrics(
    system: EvaluatedSystem, qrels: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    rows = []
    for qid, q_qrels in qrels.items():
        m = IREvaluator.evaluate_query(system.runs.get(qid, []), q_qrels)
        rows.append(
            {
                "split": system.split,
                "system": system.name,
                "query_id": qid,
                **_metrics_record(m),
            }
        )
    return rows


def _combined_per_query_ms(
    *systems: EvaluatedSystem, queries: Sequence[Query]
) -> dict[str, float]:
    """Sum per-query latencies across sequential pipeline stages."""
    out: dict[str, float] = {}
    for q in queries:
        qid = q.query_id
        parts = [s.per_query_ms.get(qid) for s in systems]
        if all(p is None for p in parts):
            continue
        out[qid] = sum(p or 0.0 for p in parts)
    return out


def _hybrid_latency(
    lexical: EvaluatedSystem, dense: EvaluatedSystem, queries: Sequence[Query]
) -> LatencyStats:
    """Combined latency of the two sequential first-stage searches."""
    return LatencyStats.from_samples(
        list(_combined_per_query_ms(lexical, dense, queries=queries).values())
    )


def _comparison_record(comp: PairedComparisonResult) -> dict[str, Any]:
    return {
        "mean_baseline_ndcg": comp.mean_baseline_ndcg,
        "mean_candidate_ndcg": comp.mean_candidate_ndcg,
        "mean_delta": comp.mean_delta,
        "ci_lower": comp.ci_lower,
        "ci_upper": comp.ci_upper,
        "wins": comp.wins,
        "ties": comp.ties,
        "losses": comp.losses,
    }


def _query_difficulty_stats(
    test_systems: Sequence[EvaluatedSystem],
    test_qrels: dict[str, dict[str, int]],
    queries: Sequence[Query],
    dense_arm_name: str,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Per-query NDCG@10 difficulty breakdown across test systems.

    Success buckets compare pure systems only: ``bm25`` against the pure
    dense encoders (``dense-*``). Hybrids and rerankers are excluded because
    they all consume BM25 candidates, which would make lexical-only success
    unreachable by construction. ``dense_arm_name`` is the frozen hybrid's
    dense arm, reported head-to-head against BM25.
    """
    names = [s.name for s in test_systems]
    ndcg: dict[str, dict[str, float]] = {}
    for s in test_systems:
        ndcg[s.name] = {
            qid: IREvaluator.evaluate_query(s.runs.get(qid, []), q_qrels).ndcg_at_10
            for qid, q_qrels in test_qrels.items()
        }
    query_text = {q.query_id: q.text for q in queries}
    dense_names = [n for n in names if n.startswith("dense-")]

    def _bucket(ids: list[str]) -> dict[str, Any]:
        return {
            "count": len(ids),
            "query_ids": ids[:max_examples],
            "examples": [
                {"query_id": qid, "query_text": query_text.get(qid, "")}
                for qid in ids[:max_examples]
            ],
        }

    all_zero: list[str] = []
    lexical_only: list[str] = []
    dense_only: list[str] = []
    bm25_only: list[str] = []
    arm_only: list[str] = []
    for qid in sorted(test_qrels):
        all_vals = [ndcg[n][qid] for n in names]
        den_vals = [ndcg[n][qid] for n in dense_names]
        bm25_val = ndcg.get("bm25", {}).get(qid, 0.0)
        arm_val = ndcg.get(dense_arm_name, {}).get(qid, 0.0)
        if all(v == 0.0 for v in all_vals):
            all_zero.append(qid)
        if bm25_val > 0.0 and all(v == 0.0 for v in den_vals):
            lexical_only.append(qid)
        if bm25_val == 0.0 and any(v > 0.0 for v in den_vals):
            dense_only.append(qid)
        if bm25_val > 0.0 and arm_val == 0.0:
            bm25_only.append(qid)
        if bm25_val == 0.0 and arm_val > 0.0:
            arm_only.append(qid)

    return {
        "n_test_queries": len(test_qrels),
        "system_groups": {
            "pure_lexical": ["bm25"],
            "pure_dense": dense_names,
        },
        "all_systems_zero_ndcg": _bucket(all_zero),
        "lexical_only_success": _bucket(lexical_only),
        "dense_only_success": _bucket(dense_only),
        "head_to_head": {
            f"bm25_vs_{dense_arm_name}": {
                "bm25_only": _bucket(bm25_only),
                f"{dense_arm_name}_only": _bucket(arm_only),
            }
        },
    }


def _package_versions() -> dict[str, str]:
    names = [
        "research-paper-semantic-search",
        "numpy",
        "pyarrow",
        "typer",
        "sentence-transformers",
        "transformers",
        "torch",
        "faiss-cpu",
        "huggingface-hub",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_report(
    path: Path,
    *,
    test_systems: Sequence[EvaluatedSystem],
    comparisons: dict[str, PairedComparisonResult],
    additional_comparisons: dict[str, dict[str, Any]],
    query_difficulty: dict[str, Any],
    frozen_system: str,
    dev_winner: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    metrics_by_system = {s.name: s.metrics for s in test_systems}
    dataset_prov = manifest.get("dataset") or {}
    corpus_dataset = dataset_prov.get("corpus_dataset", "custom")
    corpus_revision = str(dataset_prov.get("corpus_revision", "n/a"))
    lines = [
        "# SciFact Retrieval Benchmark Report",
        "",
        f"- Dataset: {corpus_dataset} ({manifest.get('dataset_name', 'scifact')}) @ "
        f"`{corpus_revision[:12]}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Hardware: {manifest.get('gpu_name', 'CPU')}",
        f"- Frozen final system: `{frozen_system}`",
        f"- Dev split: {manifest['splits']['dev_queries']} queries tuned from "
        f"{manifest['splits']['train_queries']} SciFact train queries "
        f"(fraction {manifest['config']['dev_fraction']})",
        f"- Test split: {manifest['splits']['test_queries']} BEIR 'test' queries "
        "(SciFact development claims; official test labels are not public)",
        "",
        "## Frozen configuration (selected on dev only)",
        "",
        "```json",
        json.dumps(dev_winner, indent=2, sort_keys=True),
        "```",
        "",
        "## Test metrics",
        "",
        format_ir_benchmark_table(metrics_by_system, frozen_system=frozen_system),
        "",
        "## Paired bootstrap vs BM25 (NDCG@10, 95% CI)",
        "",
        "| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, comp in comparisons.items():
        lines.append(format_paired_comparison_row(comp, f"{name} vs bm25"))
    lines += [
        "",
        "## Additional paired comparisons (NDCG@10, 95% CI)",
        "",
        "| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for rec in additional_comparisons.values():
        label = f"{rec['candidate']} vs {rec['baseline']}"
        lines.append(
            f"| `{label}` | {rec['mean_delta']:+.4f} "
            f"[{rec['ci_lower']:+.4f}, {rec['ci_upper']:+.4f}] | "
            f"{rec['wins']} | {rec['ties']} | {rec['losses']} |"
        )
    lines += [
        "",
        "## Per-system latency (test split, end-to-end ms/query)",
        "",
        "| System | Mean e2e | Median e2e | P95 e2e | Rerank-stage mean |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for system in test_systems:
        if system.latency is None:
            continue
        lat = system.latency
        rerank_mean = (
            f"{system.stage_latency.mean_ms:.2f}"
            if system.stage_latency is not None
            else "-"
        )
        lines.append(
            f"| `{system.name}` | {lat.mean_ms:.2f} | {lat.median_ms:.2f} | "
            f"{lat.p95_ms:.2f} | {rerank_mean} |"
        )
    lines += _difficulty_lines(query_difficulty)
    lines += [
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(manifest["dataset"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _difficulty_lines(query_difficulty: dict[str, Any]) -> list[str]:
    """Markdown lines for the hard-query (RQ5) breakdown."""
    lines = [
        "",
        "## Hard queries (NDCG@10 breakdown across test systems)",
        "",
        "Success buckets compare pure systems only: BM25 against the pure "
        "dense encoders; hybrids and rerankers are excluded because they all "
        "consume BM25 candidates.",
        "",
    ]

    def _bucket_lines(title: str, bucket: dict[str, Any]) -> None:
        lines.append(f"- {title}: {bucket['count']} queries")
        for ex in bucket["examples"]:
            text = ex["query_text"].replace("\n", " ")
            lines.append(f"  - `{ex['query_id']}`: \"{text}\"")

    _bucket_lines(
        "Every system scores NDCG@10 = 0",
        query_difficulty["all_systems_zero_ndcg"],
    )
    _bucket_lines(
        "BM25 scores NDCG@10 > 0 while every pure dense encoder scores 0",
        query_difficulty["lexical_only_success"],
    )
    _bucket_lines(
        "A pure dense encoder scores NDCG@10 > 0 while BM25 scores 0",
        query_difficulty["dense_only_success"],
    )
    for pair, buckets in query_difficulty["head_to_head"].items():
        for suffix, bucket in buckets.items():
            _bucket_lines(f"Head-to-head {pair}: `{suffix}` scores > 0 only", bucket)
    return lines


def run_benchmark(
    out_dir: Path,
    *,
    dataset: BenchmarkDataset | None = None,
    dense_specs: Sequence[DenseModelSpec] = DENSE_MODEL_SPECS,
    rerank_depths: Sequence[int] = (20, 50, 100),
    dev_fraction: float = 0.2,
    k_eval: int = 100,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
    rrf_k_grid: Sequence[float] = (10.0, 30.0, 60.0, 100.0),
    rrf_weight_grid: Sequence[tuple[float, float]] = (
        (1.0, 1.0),
        (1.5, 1.0),
        (1.0, 1.5),
        (2.0, 1.0),
        (1.0, 2.0),
    ),
    dense_backends: Sequence[str] = ("exact", "faiss"),
    encoder_factory: EncoderFactory | None = None,
    scorer_factory: ScorerFactory | None = None,
    git_sha: str = "unknown",
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full benchmark and write artifacts to ``out_dir``.

    Returns a JSON-serializable summary. Neural components are built through
    injectable factories so tests can substitute deterministic fakes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    stage_times: dict[str, float] = {}
    encoder_factory = encoder_factory or _default_encoder_factory
    scorer_factory = scorer_factory or _default_scorer_factory

    # ---------------- data + splits ----------------
    t0 = time.perf_counter()
    data = dataset if dataset is not None else load_scifact_benchmark()
    stage_times["load_dataset_seconds"] = time.perf_counter() - t0

    docs = data.documents
    test_queries = [
        q for q in data.queries if q.split == "test" and q.query_id in data.qrels
    ]
    train_queries = [
        q for q in data.queries if q.split == "train" and q.query_id in data.qrels
    ]
    if not train_queries:
        raise ValueError("Benchmark dataset has no judged train queries for dev split.")
    dev_ids = make_dev_split(
        [q.query_id for q in train_queries], fraction=dev_fraction
    )
    dev_queries = [q for q in train_queries if q.query_id in dev_ids]
    test_qrels = {q.query_id: data.qrels[q.query_id] for q in test_queries}
    dev_qrels = {q.query_id: data.qrels[q.query_id] for q in dev_queries}

    # ---------------- first-stage indexes ----------------
    t0 = time.perf_counter()
    bm25 = BM25Retriever().index(docs)
    stage_times["bm25_index_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tfidf = HashedTfidfRetriever().index(docs)
    stage_times["tfidf_index_seconds"] = time.perf_counter() - t0

    dense_retrievers: dict[str, SentenceTransformerRetriever] = {}
    dense_info: dict[str, dict[str, Any]] = {}
    for spec in dense_specs:
        encoder = encoder_factory(spec)
        retriever = SentenceTransformerRetriever(
            encoder=encoder, backends=dense_backends, name=spec.key
        ).index(docs)
        dense_retrievers[spec.key] = retriever
        dense_info[spec.key] = retriever.encoder_info()
    stage_times["dense_index_seconds"] = sum(
        info["encode_seconds"] + info["index_seconds"] for info in dense_info.values()
    )

    # ---------------- dev evaluation ----------------
    t0 = time.perf_counter()
    dev_systems: list[EvaluatedSystem] = []
    dev_systems.append(
        _evaluate_retriever("bm25", "dev", bm25, dev_queries, dev_qrels, k_eval)
    )
    dev_systems.append(
        _evaluate_retriever("tfidf", "dev", tfidf, dev_queries, dev_qrels, k_eval)
    )
    bm25_dev = dev_systems[0]
    dense_dev_runs: dict[str, dict[str, list[SearchResult]]] = {}
    for spec in dense_specs:
        system = _evaluate_retriever(
            f"dense-{spec.key}",
            "dev",
            dense_retrievers[spec.key],
            dev_queries,
            dev_qrels,
            k_eval,
            config=dense_model_to_dict(spec),
            search_kwargs={"backend": "exact"},
        )
        dev_systems.append(system)
        dense_dev_runs[spec.key] = system.runs

    # Hybrid grid over cached first-stage dev runs.
    grid_results: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float, float, int] | None = None
    best_config: dict[str, Any] | None = None
    best_dev_runs: dict[str, list[SearchResult]] | None = None
    for spec_index, spec in enumerate(dense_specs):
        for k_rrf in rrf_k_grid:
            for lexical_weight, dense_weight in rrf_weight_grid:
                fused = _fuse_runs(
                    bm25_dev.runs,
                    dense_dev_runs[spec.key],
                    dev_queries,
                    k_rrf=k_rrf,
                    lexical_weight=lexical_weight,
                    dense_weight=dense_weight,
                    k=k_eval,
                )
                metrics = IREvaluator.evaluate_run(fused, dev_qrels)
                grid_results.append(
                    {
                        "dense_model": spec.key,
                        "k_rrf": k_rrf,
                        "lexical_weight": lexical_weight,
                        "dense_weight": dense_weight,
                        "ndcg_at_10": metrics.ndcg_at_10,
                        "recall_at_100": metrics.recall_at_100,
                        "map_at_100": metrics.map_at_100,
                    }
                )
                # Deterministic selection: quality first, then simpler config.
                key = (
                    -metrics.ndcg_at_10,
                    -metrics.recall_at_100,
                    k_rrf,
                    lexical_weight,
                    dense_weight,
                    spec_index,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_config = {
                        "dense_model": spec.key,
                        "k_rrf": k_rrf,
                        "lexical_weight": lexical_weight,
                        "dense_weight": dense_weight,
                    }
                    best_dev_runs = fused
    assert best_config is not None and best_dev_runs is not None
    best_spec = next(s for s in dense_specs if s.key == best_config["dense_model"])

    # Rerank selection on dev uses the frozen hybrid candidates; "no rerank"
    # competes with every CE depth so the frozen system is chosen honestly.
    pair_scorer = scorer_factory()
    ce_reranker = CrossEncoderReranker(scorer=pair_scorer)
    ce_reranker.index_documents(docs)
    lexical_reranker = LexicalOverlapReranker()
    lexical_reranker.index_documents(docs)

    dense_dev_best = _dense_dev_system(dev_systems, best_spec.key)
    dev_first_stage_ms = _combined_per_query_ms(
        bm25_dev, dense_dev_best, queries=dev_queries
    )
    hybrid_dev_system = EvaluatedSystem(
        name="hybrid-rrf",
        split="dev",
        runs=best_dev_runs,
        metrics=IREvaluator.evaluate_run(best_dev_runs, dev_qrels),
        latency=_hybrid_latency(bm25_dev, dense_dev_best, dev_queries),
        config=best_config,
    )
    dev_systems.append(hybrid_dev_system)

    ce_dev_by_depth: dict[int, EvaluatedSystem] = {}
    for depth in sorted(rerank_depths):
        system = _evaluate_reranker(
            f"hybrid+ce-rerank@{depth}",
            "dev",
            ce_reranker,
            best_dev_runs,
            dev_queries,
            dev_qrels,
            k_eval,
            candidate_depth=min(depth, k_eval),
            first_stage_ms=dev_first_stage_ms,
        )
        dev_systems.append(system)
        ce_dev_by_depth[depth] = system

    # Prefer no rerank, then smaller depth, on quality ties.
    rerank_selection: list[dict[str, Any]] = [
        {
            "system": "hybrid-rrf",
            "depth": None,
            "ndcg_at_10": hybrid_dev_system.metrics.ndcg_at_10,
            "mean_latency_ms": (
                hybrid_dev_system.latency.mean_ms
                if hybrid_dev_system.latency
                else None
            ),
        }
    ]
    best_depth: int | None = None
    best_depth_ndcg = hybrid_dev_system.metrics.ndcg_at_10
    for depth in sorted(rerank_depths):
        system = ce_dev_by_depth[depth]
        rerank_selection.append(
            {
                "system": system.name,
                "depth": depth,
                "ndcg_at_10": system.metrics.ndcg_at_10,
                "mean_latency_ms": system.latency.mean_ms if system.latency else None,
            }
        )
        if system.metrics.ndcg_at_10 > best_depth_ndcg + 1e-12:
            best_depth_ndcg = system.metrics.ndcg_at_10
            best_depth = depth

    dev_lexical = _evaluate_reranker(
        "hybrid+lexical-rerank",
        "dev",
        lexical_reranker,
        best_dev_runs,
        dev_queries,
        dev_qrels,
        k_eval,
        candidate_depth=k_eval,
        first_stage_ms=dev_first_stage_ms,
    )
    dev_systems.append(dev_lexical)
    stage_times["dev_evaluation_seconds"] = time.perf_counter() - t0

    # ---------------- frozen test evaluation ----------------
    t0 = time.perf_counter()
    test_systems: list[EvaluatedSystem] = []
    test_systems.append(
        _evaluate_retriever("bm25", "test", bm25, test_queries, test_qrels, k_eval)
    )
    test_systems.append(
        _evaluate_retriever("tfidf", "test", tfidf, test_queries, test_qrels, k_eval)
    )
    bm25_test = test_systems[0]
    test_dense_systems: dict[str, EvaluatedSystem] = {}
    for spec in dense_specs:
        system = _evaluate_retriever(
            f"dense-{spec.key}",
            "test",
            dense_retrievers[spec.key],
            test_queries,
            test_qrels,
            k_eval,
            config=dense_model_to_dict(spec),
            search_kwargs={"backend": "exact"},
        )
        test_systems.append(system)
        test_dense_systems[spec.key] = system

    # Frozen hybrid from cached test runs of its two arms.
    best_dense_test = test_dense_systems[best_spec.key]
    hybrid_test_runs = _fuse_runs(
        bm25_test.runs,
        best_dense_test.runs,
        test_queries,
        k_rrf=float(best_config["k_rrf"]),
        lexical_weight=float(best_config["lexical_weight"]),
        dense_weight=float(best_config["dense_weight"]),
        k=k_eval,
    )
    hybrid_test = EvaluatedSystem(
        name="hybrid-rrf",
        split="test",
        runs=hybrid_test_runs,
        metrics=IREvaluator.evaluate_run(hybrid_test_runs, test_qrels),
        latency=_hybrid_latency(bm25_test, best_dense_test, test_queries),
        config=best_config,
    )
    test_systems.append(hybrid_test)

    test_first_stage_ms = _combined_per_query_ms(
        bm25_test, best_dense_test, queries=test_queries
    )
    test_systems.append(
        _evaluate_reranker(
            "hybrid+lexical-rerank",
            "test",
            lexical_reranker,
            hybrid_test_runs,
            test_queries,
            test_qrels,
            k_eval,
            candidate_depth=k_eval,
            first_stage_ms=test_first_stage_ms,
        )
    )
    for depth in sorted(rerank_depths):
        test_systems.append(
            _evaluate_reranker(
                f"hybrid+ce-rerank@{depth}",
                "test",
                ce_reranker,
                hybrid_test_runs,
                test_queries,
                test_qrels,
                k_eval,
                candidate_depth=min(depth, k_eval),
                first_stage_ms=test_first_stage_ms,
            )
        )
    stage_times["test_evaluation_seconds"] = time.perf_counter() - t0

    # ---------------- statistics ----------------
    t0 = time.perf_counter()
    comparisons: dict[str, PairedComparisonResult] = {}
    per_query_deltas: dict[str, dict[str, float]] = {}
    base_ndcg = {
        qid: IREvaluator.evaluate_query(bm25_test.runs.get(qid, []), q_qrels).ndcg_at_10
        for qid, q_qrels in test_qrels.items()
    }
    for system in test_systems:
        if system.name == "bm25":
            continue
        comp = compare_systems_bootstrap(
            bm25_test.runs,
            system.runs,
            test_qrels,
            n_bootstrap=n_bootstrap,
            random_state=bootstrap_seed,
        )
        comparisons[system.name] = comp
        per_query_deltas[system.name] = {
            qid: round(
                IREvaluator.evaluate_query(system.runs.get(qid, []), q_qrels).ndcg_at_10
                - base_ndcg[qid],
                6,
            )
            for qid, q_qrels in test_qrels.items()
        }

    # Additional paired comparisons: RQ3 hybrid vs its dense arm, and RQ4 each
    # CE depth vs the frozen hybrid (kept as test ablations either way).
    additional_specs: list[tuple[EvaluatedSystem, EvaluatedSystem]] = [
        (hybrid_test, test_dense_systems[best_spec.key]),
    ]
    for depth in sorted(rerank_depths):
        name = f"hybrid+ce-rerank@{depth}"
        candidate = next(s for s in test_systems if s.name == name)
        additional_specs.append((candidate, hybrid_test))
    additional_comparisons: dict[str, dict[str, Any]] = {}
    for candidate, baseline in additional_specs:
        comp = compare_systems_bootstrap(
            baseline.runs,
            candidate.runs,
            test_qrels,
            n_bootstrap=n_bootstrap,
            random_state=bootstrap_seed,
        )
        comp_key = f"{candidate.name}_vs_{baseline.name}"
        additional_comparisons[comp_key] = {
            "candidate": candidate.name,
            "baseline": baseline.name,
            **_comparison_record(comp),
        }
    query_difficulty = _query_difficulty_stats(
        test_systems,
        test_qrels,
        test_queries,
        dense_arm_name=f"dense-{best_spec.key}",
    )
    stage_times["bootstrap_seconds"] = time.perf_counter() - t0

    # ---------------- FAISS parity check ----------------
    faiss_parity: dict[str, dict[str, Any]] = {}
    if "faiss" in dense_backends:
        parity_queries = dev_queries[: max(1, min(len(dev_queries), 50))]
        for spec in dense_specs:
            retriever = dense_retrievers[spec.key]
            mismatches = 0
            checked = 0
            for query in parity_queries:
                exact = retriever.search(query.text, k=k_eval, backend="exact")
                approx = retriever.search(query.text, k=k_eval, backend="faiss")
                checked += 1
                if [r.doc_id for r in exact] != [r.doc_id for r in approx]:
                    mismatches += 1
            faiss_parity[spec.key] = {
                "queries_checked": checked,
                "ranking_mismatches": mismatches,
            }

    # ---------------- manifest + outputs ----------------
    frozen_system_name = (
        "hybrid-rrf" if best_depth is None else f"hybrid+ce-rerank@{best_depth}"
    )
    frozen_config = {
        "dense_model": best_spec.key,
        "rrf": {
            "k": best_config["k_rrf"],
            "lexical_weight": best_config["lexical_weight"],
            "dense_weight": best_config["dense_weight"],
        },
        "rerank_depth": best_depth,
        "k_eval": k_eval,
        "dev_fraction": dev_fraction,
        "dev_split_salt": "paper-search-dev-v1",
        "seeds": {"bootstrap": bootstrap_seed},
        "n_bootstrap": n_bootstrap,
    }
    frozen_dev_ndcg = (
        hybrid_dev_system.metrics.ndcg_at_10
        if best_depth is None
        else ce_dev_by_depth[best_depth].metrics.ndcg_at_10
    )
    dev_winner = {
        "frozen_config": frozen_config,
        "frozen_system": frozen_system_name,
        "dev_ndcg_at_10": frozen_dev_ndcg,
        "grid_size": len(grid_results),
        "rerank_selection_dev": rerank_selection,
        "top_grid_configs": sorted(
            grid_results, key=lambda r: -r["ndcg_at_10"]
        )[:5],
    }

    manifest: dict[str, Any] = {
        "git_sha": git_sha,
        "package_version": paper_search.__version__,
        "dataset_name": data.name,
        "dataset": data.provenance,
        "splits": {
            "train_queries": len(train_queries),
            "dev_queries": len(dev_queries),
            "test_queries": len(test_queries),
            "dev_query_ids_sha256": _ids_digest([q.query_id for q in dev_queries]),
        },
        "config": frozen_config,
        "frozen_system": frozen_system_name,
        "dev_rerank_selection": rerank_selection,
        "models": {
            "dense": {spec.key: dense_model_to_dict(spec) for spec in dense_specs},
            "reranker": reranker_model_to_dict(),
        },
        "encoder_info": dense_info,
        "faiss_parity": faiss_parity,
        "stage_times_seconds": stage_times,
        "total_seconds": time.perf_counter() - t_start,
        "package_versions": _package_versions(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    # metrics.json
    metrics_payload = {
        "dataset": data.name,
        "splits": manifest["splits"],
        "config": frozen_config,
        "frozen_system": frozen_system_name,
        "dev_systems": {
            s.name: {**_metrics_record(s.metrics), "config": s.config}
            for s in dev_systems
        },
        "test_systems": {
            s.name: {
                **_metrics_record(s.metrics),
                "config": s.config,
                "latency": s.latency.to_dict() if s.latency else None,
                "rerank_latency": (
                    s.stage_latency.to_dict() if s.stage_latency else None
                ),
            }
            for s in test_systems
        },
        "comparisons_vs_bm25": {
            name: _comparison_record(c) for name, c in comparisons.items()
        },
        "additional_comparisons": additional_comparisons,
        "per_query_ndcg_deltas_vs_bm25": per_query_deltas,
        "query_difficulty": query_difficulty,
        "dev_tuning": dev_winner,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )

    # runs.parquet + per_query_metrics.parquet
    run_rows: list[dict[str, Any]] = []
    pq_rows: list[dict[str, Any]] = []
    for system in [*dev_systems, *test_systems]:
        qrels = dev_qrels if system.split == "dev" else test_qrels
        for qid, results in system.runs.items():
            for res in results:
                run_rows.append(
                    {
                        "split": system.split,
                        "system": system.name,
                        "query_id": qid,
                        "rank": res.rank,
                        "doc_id": res.doc_id,
                        "score": res.score,
                    }
                )
        pq_rows.extend(_per_query_metrics(system, qrels))
    _write_parquet(out_dir / "runs.parquet", run_rows)
    _write_parquet(out_dir / "per_query_metrics.parquet", pq_rows)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        out_dir / "report.md",
        test_systems=test_systems,
        comparisons=comparisons,
        additional_comparisons=additional_comparisons,
        query_difficulty=query_difficulty,
        frozen_system=frozen_system_name,
        dev_winner=dev_winner,
        manifest=manifest,
    )

    return {
        "out_dir": str(out_dir),
        "frozen_config": frozen_config,
        "frozen_system": frozen_system_name,
        "test_metrics": {
            s.name: _metrics_record(s.metrics) for s in test_systems
        },
        "manifest": manifest,
    }


def _dense_dev_system(
    dev_systems: Sequence[EvaluatedSystem], key: str
) -> EvaluatedSystem:
    for system in dev_systems:
        if system.name == f"dense-{key}":
            return system
    raise KeyError(f"No dev system recorded for dense model {key}")


def _ids_digest(query_ids: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for qid in sorted(query_ids):
        digest.update(qid.encode())
        digest.update(b"\n")
    return digest.hexdigest()
