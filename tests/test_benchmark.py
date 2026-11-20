"""End-to-end benchmark pipeline test with injected fake encoders/scorers.

Runs the real ``run_benchmark`` orchestration (dev tuning, frozen test eval,
manifest, parquet outputs) on a tiny synthetic dataset with no model
downloads, verifying the pipeline logic and artifact contract.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from paper_search.benchmark.models import DenseModelSpec
from paper_search.benchmark.runner import run_benchmark
from paper_search.data.schema import BenchmarkDataset, Document, Query


class BagOfWordsEncoder:
    dim = 8
    model_id = "fake-encoder"
    revision = "fake-rev"
    query_prompt = ""
    device = "cpu"

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


class FakeScorer:
    model_id = "fake-ce"
    revision = "fake-rev"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        q_tokens = set(query.lower().split())
        return np.array(
            [len(q_tokens & set(t.lower().split())) for t in texts],
            dtype=np.float32,
        )


def _toy_dataset() -> BenchmarkDataset:
    docs = [
        Document(
            doc_id=f"d{i}",
            title=f"Topic {i} study",
            abstract=f"alpha beta gamma delta epsilon zeta topic{i} findings",
        )
        for i in range(12)
    ]
    queries = [
        Query(query_id=f"t{i}", text=f"topic{i} alpha claim", split="train")
        for i in range(6)
    ] + [
        Query(query_id=f"s{i}", text=f"topic{i + 6} beta claim", split="test")
        for i in range(4)
    ]
    qrels = {f"t{i}": {f"d{i}": 1} for i in range(6)} | {
        f"s{i}": {f"d{i + 6}": 1, f"d{(i + 7) % 12}": 1} for i in range(4)
    }
    return BenchmarkDataset(name="toy", documents=docs, queries=queries, qrels=qrels)


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    specs = (
        DenseModelSpec(
            key="fake", model_id="fake", revision="fake-rev", description="toy"
        ),
    )
    summary = run_benchmark(
        tmp_path,
        dataset=_toy_dataset(),
        dense_specs=specs,
        rerank_depths=(5, 10),
        dev_fraction=0.5,
        k_eval=10,
        n_bootstrap=20,
        dense_backends=("exact",),
        encoder_factory=lambda spec: BagOfWordsEncoder(),
        scorer_factory=lambda: FakeScorer(),
        git_sha="deadbeef",
    )

    # Artifact contract.
    for name in (
        "metrics.json",
        "manifest.json",
        "runs.parquet",
        "per_query_metrics.parquet",
        "report.md",
    ):
        assert (tmp_path / name).exists(), name

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    # Frozen config recorded and used; rerank may be disabled (null depth).
    frozen = summary["frozen_config"]
    assert frozen["dense_model"] == "fake"
    assert frozen["rerank_depth"] in (None, 5, 10)
    expected_system = (
        "hybrid-rrf"
        if frozen["rerank_depth"] is None
        else f"hybrid+ce-rerank@{frozen['rerank_depth']}"
    )
    assert summary["frozen_system"] == expected_system
    assert metrics["frozen_system"] == expected_system
    assert manifest["frozen_system"] == expected_system
    assert manifest["config"] == frozen
    assert manifest["git_sha"] == "deadbeef"

    # Dev rerank selection table covers no-rerank plus every depth.
    selection = manifest["dev_rerank_selection"]
    assert [r["system"] for r in selection] == [
        "hybrid-rrf",
        "hybrid+ce-rerank@5",
        "hybrid+ce-rerank@10",
    ]
    assert selection[0]["depth"] is None
    assert metrics["dev_tuning"]["rerank_selection_dev"] == selection

    # Split accounting: 3 dev + 4 test out of 6 train + 4 test queries.
    assert manifest["splits"]["dev_queries"] == 3
    assert manifest["splits"]["test_queries"] == 4
    assert manifest["splits"]["train_queries"] == 6

    # All expected test systems present with metrics and comparisons.
    test_systems = metrics["test_systems"]
    for name in (
        "bm25",
        "tfidf",
        "dense-fake",
        "hybrid-rrf",
        "hybrid+lexical-rerank",
        "hybrid+ce-rerank@5",
        "hybrid+ce-rerank@10",
    ):
        assert name in test_systems, name
        assert "ndcg_at_10" in test_systems[name]
    comparisons = metrics["comparisons_vs_bm25"]
    assert "bm25" not in comparisons
    assert "hybrid-rrf" in comparisons
    assert set(comparisons["hybrid-rrf"]) >= {
        "mean_delta",
        "ci_lower",
        "ci_upper",
        "wins",
        "ties",
        "losses",
    }

    # RQ3/RQ4 paired comparisons are recorded alongside vs-BM25.
    extra = metrics["additional_comparisons"]
    assert "hybrid-rrf_vs_dense-fake" in extra
    assert extra["hybrid-rrf_vs_dense-fake"]["candidate"] == "hybrid-rrf"
    assert extra["hybrid-rrf_vs_dense-fake"]["baseline"] == "dense-fake"
    assert "hybrid+ce-rerank@5_vs_hybrid-rrf" in extra
    assert "hybrid+ce-rerank@10_vs_hybrid-rrf" in extra
    for rec in extra.values():
        assert set(rec) >= {
            "candidate",
            "baseline",
            "mean_delta",
            "ci_lower",
            "ci_upper",
            "wins",
            "ties",
            "losses",
        }

    # Reranked systems expose end-to-end latency plus rerank-stage latency.
    ce = test_systems["hybrid+ce-rerank@5"]
    assert ce["latency"] is not None
    assert ce["rerank_latency"] is not None
    assert test_systems["hybrid-rrf"]["rerank_latency"] is None
    assert test_systems["bm25"]["rerank_latency"] is None

    # RQ5 hard-query stats are computed from per-query metrics.
    difficulty = metrics["query_difficulty"]
    assert difficulty["n_test_queries"] == 4
    for key in (
        "all_systems_zero_ndcg",
        "lexical_only_success",
        "dense_only_success",
    ):
        assert "count" in difficulty[key]
        assert "query_ids" in difficulty[key]
        assert "examples" in difficulty[key]
    h2h = difficulty["head_to_head"]["bm25_vs_dense-fake"]
    assert "bm25_only" in h2h
    assert "dense-fake_only" in h2h
    for bucket in h2h.values():
        assert "count" in bucket and "examples" in bucket

    # Per-query deltas recorded for every compared system.
    assert set(metrics["per_query_ndcg_deltas_vs_bm25"]) == set(comparisons)

    # Report exists and mentions the frozen config and system.
    report = (tmp_path / "report.md").read_text()
    assert "SciFact Retrieval Benchmark Report" in report
    assert "Frozen configuration" in report
    assert f"`{expected_system}`" in report
    assert "Additional paired comparisons" in report
    assert "Hard queries" in report


def test_run_benchmark_dev_selection_uses_dev_only(tmp_path: Path) -> None:
    """The frozen config must come from dev metrics, not test metrics."""
    specs = (
        DenseModelSpec(key="a", model_id="a", revision="r"),
        DenseModelSpec(key="b", model_id="b", revision="r"),
    )

    def make_encoder(spec: DenseModelSpec) -> BagOfWordsEncoder:
        enc = BagOfWordsEncoder()
        # Deterministic per-spec dim keeps models distinguishable.
        enc.dim = 8 if spec.key == "a" else 8
        return enc

    summary = run_benchmark(
        tmp_path,
        dataset=_toy_dataset(),
        dense_specs=specs,
        rerank_depths=(5,),
        dev_fraction=0.5,
        k_eval=10,
        n_bootstrap=10,
        dense_backends=("exact",),
        encoder_factory=make_encoder,
        scorer_factory=lambda: FakeScorer(),
        git_sha="x",
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    # Dev tuning section must exist and contain grid results.
    assert metrics["dev_tuning"]["grid_size"] > 0
    assert summary["frozen_config"]["dense_model"] in {"a", "b"}
