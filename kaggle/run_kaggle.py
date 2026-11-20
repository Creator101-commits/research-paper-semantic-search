"""Kaggle entrypoint for the SciFact retrieval benchmark.

Installs ``paper_search`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh`` (the preserved repo tree under ``/kaggle/input``),
runs the full dev-tuned benchmark, cross-checks metrics against
``pytrec_eval``, and writes all artifacts under ``/kaggle/working/``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    """Locate the project root (pyproject.toml + src/paper_search) under /kaggle/input."""
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "paper_search").is_dir():
            return candidate.parent
    return None


def _install_package() -> None:
    """Install ``paper_search`` from the preserved source-dataset tree.

    ``scripts/kaggle_run.sh`` uploads ``git archive HEAD`` output, so the
    dataset mount contains the repo tree (``pyproject.toml`` plus
    ``src/paper_search``).
    """
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/paper_search)."
        )

    # /kaggle/input is read-only; stage a writable copy for the build.
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "sentence-transformers>=3.0",
            "faiss-cpu>=1.8",
            "huggingface-hub>=0.26",
            # Pin: Kaggle images may ship an older pytrec_eval that pip
            # would otherwise keep as "already satisfied".
            "pytrec_eval==0.5",
        ]
    )


def _git_sha() -> str:
    proj_root = _find_project_root()
    info = proj_root / "run_info.json" if proj_root is not None else None
    if info is not None and info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _dataset_sources() -> list[str]:
    """Read dataset_sources from the packaged kernel metadata, if present."""
    proj_root = _find_project_root()
    meta = (
        proj_root / "kaggle" / "kernel-metadata.json"
        if proj_root is not None
        else None
    )
    if meta is not None and meta.exists():
        try:
            return list(json.loads(meta.read_text()).get("dataset_sources", []))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def _package_versions() -> dict[str, str]:
    names = [
        "research-paper-semantic-search",
        "paper-search",
        "numpy",
        "pyarrow",
        "typer",
        "sentence-transformers",
        "transformers",
        "torch",
        "faiss-cpu",
        "huggingface-hub",
        "pytrec_eval",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _load_test_qrels() -> dict[str, dict[str, int]]:
    """Reload SciFact (HF cache hit) to recover the test qrels for cross-check."""
    from paper_search.data.scifact import load_scifact_benchmark

    data = load_scifact_benchmark()
    return {
        q.query_id: data.qrels[q.query_id]
        for q in data.queries
        if q.split == "test" and q.query_id in data.qrels
    }


def _runs_from_parquet(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return {system: {query_id: {doc_id: score}}} for test-split rows.

    Rows are ordered by the run's explicit ``rank`` column and scores are
    rewritten to a strictly decreasing sequence. pytrec_eval derives each
    run's ranking from scores alone (ties by descending docno), while our
    artifacts carry rank order with doc_id-ascending ties; rank-ordered
    scores make the cross-check compare metric math on identical rankings.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["split", "system", "query_id", "rank", "doc_id", "score"],
    )
    rows = [r for r in table.to_pylist() if r["split"] == "test"]
    rows.sort(key=lambda r: (r["system"], r["query_id"], r["rank"]))

    ordered: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        ordered.setdefault(row["system"], {}).setdefault(row["query_id"], []).append(
            row["doc_id"]
        )

    runs: dict[str, dict[str, dict[str, float]]] = {
        system: {
            qid: {doc_id: float(len(docs) - i) for i, doc_id in enumerate(docs)}
            for qid, docs in qmap.items()
        }
        for system, qmap in ordered.items()
    }
    return runs


# Our metric -> (pytrec_eval measure, output key). All measures are requested
# unparameterized so trec_eval emits its default cutoff ladder (5..1000),
# which includes every cutoff we need; parameterized forms like "recall.10,100"
# lost their second cutoff on Kaggle's pytrec_eval build. recip_rank is skipped:
# our mrr_at_10 is a top-10 cutoff while pytrec_eval's recip_rank scans the full run.
_METRIC_MAP = {
    "ndcg_at_10": ("ndcg_cut", "ndcg_cut_10"),
    "map_at_100": ("map", "map"),
    "recall_at_10": ("recall", "recall_10"),
    "recall_at_100": ("recall", "recall_100"),
    "precision_at_10": ("P", "P_10"),
}


def _our_per_query_metrics(
    pq_parquet: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {system: {query_id: {metric: value}}} for test-split rows."""
    import pyarrow.parquet as pq

    metric_cols = ["ndcg_at_10", "map_at_100", "recall_at_10", "recall_at_100", "precision_at_10"]
    table = pq.read_table(
        pq_parquet,
        columns=["split", "system", "query_id", *metric_cols],
    )
    ours: dict[str, dict[str, dict[str, float]]] = {}
    for row in table.to_pylist():
        if row["split"] != "test":
            continue
        ours.setdefault(row["system"], {})[row["query_id"]] = {
            col: float(row[col]) for col in metric_cols
        }
    return ours


def _pytrec_crosscheck(
    runs_parquet: Path,
    pq_parquet: Path,
    test_qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Compare internal metrics with pytrec_eval; returns per-system mean deltas.

    Means are computed over the intersection of query IDs that pytrec_eval
    evaluated and that our per-query table covers, so the comparison is
    like-for-like even if a query is absent from one side.
    """
    import pytrec_eval

    try:
        print(f"pytrec_eval {metadata.version('pytrec_eval')}", flush=True)
    except metadata.PackageNotFoundError:
        print("pytrec_eval version unknown", flush=True)
    runs = _runs_from_parquet(runs_parquet)
    qrels_pt = {
        qid: {did: int(rel) for did, rel in rels.items()}
        for qid, rels in test_qrels.items()
    }
    wanted = {measure for measure, _ in _METRIC_MAP.values()}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_pt, wanted)

    ours_pq = _our_per_query_metrics(pq_parquet)

    report: dict[str, dict[str, float]] = {}
    worst = 0.0
    for system, run in runs.items():
        per_query = evaluator.evaluate(run)
        expected_keys = {out_key for _, out_key in _METRIC_MAP.values()}
        for qid, pt_metrics in per_query.items():
            missing = expected_keys - set(pt_metrics)
            if missing:
                raise AssertionError(
                    f"pytrec_eval query {qid} (system {system}) missing measures "
                    f"{sorted(missing)}; got {sorted(pt_metrics)}"
                )
        our_rows = ours_pq.get(system, {})
        means: dict[str, float] = {}
        for our_name, (_, out_key) in _METRIC_MAP.items():
            deltas = []
            for qid, pt_metrics in per_query.items():
                if qid not in our_rows:
                    continue
                pt_val = float(pt_metrics[out_key])
                our_val = our_rows[qid][our_name]
                deltas.append(abs(pt_val - our_val))
            if not deltas:
                raise AssertionError(f"No overlapping queries for system {system}")
            means[out_key] = sum(deltas) / len(deltas)
            worst = max(worst, means[out_key])
        report[system] = means

    print(f"pytrec_eval cross-check worst |delta| = {worst:.3e}", flush=True)
    if worst > 1e-6:
        raise AssertionError(
            f"Metric disagreement with pytrec_eval exceeds 1e-6: {report}"
        )
    (WORK / "pytrec_crosscheck.json").write_text(
        json.dumps({"max_abs_delta": worst, "per_system_mean_deltas": report}, indent=2)
        + "\n"
    )
    return report


def main() -> None:
    t0 = time.time()
    _install_package()

    git_sha = _git_sha()
    gpu_name = _gpu_name()
    print(f"git_sha={git_sha} gpu={gpu_name}", flush=True)

    from paper_search.benchmark.runner import run_benchmark

    summary = run_benchmark(
        BENCH,
        git_sha=git_sha,
        extra_manifest={
            "gpu_name": gpu_name,
            "python_version": sys.version.split()[0],
            "kaggle_dataset_sources": _dataset_sources(),
        },
    )

    # FAISS parity must be exact for every dense model.
    parity = summary["manifest"]["faiss_parity"]
    bad = {k: v for k, v in parity.items() if v["ranking_mismatches"] != 0}
    if bad:
        raise AssertionError(f"FAISS/exact ranking mismatches: {bad}")
    print("FAISS parity: all dense models match exact search", flush=True)

    test_qrels = _load_test_qrels()
    _pytrec_crosscheck(BENCH / "runs.parquet", BENCH / "per_query_metrics.parquet", test_qrels)

    env = {
        "git_sha": git_sha,
        "gpu_name": gpu_name,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "wall_seconds_total": time.time() - t0,
    }
    (WORK / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    frozen = summary["frozen_config"]
    print("\n=== FROZEN SYSTEM ===")
    print(summary["frozen_system"])
    print("\n=== FROZEN CONFIG ===")
    print(json.dumps(frozen, indent=2))
    print("\n=== TEST METRICS (NDCG@10) ===")
    for name, m in sorted(
        summary["test_metrics"].items(), key=lambda kv: -kv[1]["ndcg_at_10"]
    ):
        print(f"{name:28s} {m['ndcg_at_10']:.4f}")
    print(f"\nArtifacts written under {BENCH} and {WORK}", flush=True)


if __name__ == "__main__":
    main()
