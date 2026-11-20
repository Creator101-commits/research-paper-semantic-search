# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, benchmark pipeline, Kaggle scripts, README claims, and figure provenance.

## Verified locally

- `uv run pytest`: 44 passed, 1 skipped (FAISS parity test skips when faiss-cpu is not installed locally; it runs on Kaggle).
- `uv run ruff check .`: clean.
- `uv run mypy src`: clean under strict mode.
- `uv sync --locked --all-groups`: lockfile is current; the `neural` extra stays out of CI installs.
- Tests cover schema validation, the offline fixture loader, SciFact parsing rules on synthetic parquet/qrels, deterministic dev splitting, BM25/TF-IDF/hybrid ranking, injected-encoder dense retrieval, FAISS/exact parity logic, reranker candidate handling, CLI commands, bootstrap statistics, and the benchmark runner's dev-tuning/frozen-test flow. No test downloads models or real data.

## Verified on Kaggle

- Kernel `sreeharshak101/research-paper-semantic-search-scifact-benchmark`, version 13, git SHA `c66cf51`, Tesla T4.
- Real BEIR SciFact download from `BeIR/scifact` and `BeIR/scifact-qrels` at pinned revisions; record counts and qrel foreign keys validated (5,183 docs, 1,109 queries, 809 train / 300 test judged queries).
- Three sentence-transformer bi-encoders encoded on GPU; FAISS `IndexFlatIP` rankings matched exact NumPy inner-product search on 50 dev queries per model with zero mismatches.
- Dev-only tuning selected `bge-base` + weighted RRF (k=10, lexical 1.0 / dense 2.0) over a 60-config grid, then chose the rerank option among {no rerank, CE depth 20, 50, 100}: no rerank won (dev NDCG@10 0.7674 vs 0.7138/0.6962/0.6949), so the frozen system is `hybrid-rrf` with `rerank_depth: null`. The frozen config was evaluated once on the 300-query test split; CE depths appear on test only as labeled ablations.
- Reranked-system latency is reported end-to-end (first-stage hybrid plus rerank); the rerank-stage-only mean is kept separately in `metrics.json` (`rerank_latency`) and `report.md`.
- Per-query difficulty buckets in `metrics.json.query_difficulty` compare pure systems only: BM25 against the three `dense-*` encoders. Hybrids and rerankers are excluded because they all consume BM25 candidates, which would make lexical-only success structurally impossible. A head-to-head `bm25` vs `dense-bge-base` breakdown is reported alongside.
- `pytrec_eval` 0.5 cross-check: per-query NDCG@10, MAP@100, Recall@10, Recall@100, and Precision@10 agree exactly (max abs delta 0.0) with the internal evaluator across all 10 systems and 300 test queries (see `results/pytrec_crosscheck.json`). Runs are fed to pytrec_eval as rank-ordered decreasing scores because it derives rankings from scores alone; this validates metric math on identical rankings. MRR@10 is not cross-checked because pytrec_eval's `recip_rank` scans the full run while ours is a top-10 cutoff.
- `scripts/kaggle_run.sh` derives the kernel id and source dataset id from `kaggle/kernel-metadata.json`, exits nonzero on kernel ERROR/CANCEL or poll timeout after downloading whatever output exists, and fails if the downloaded run manifest's `git_sha` differs from `HEAD`.
- Full artifacts (metrics.json, manifest.json, runs.parquet, per_query_metrics.parquet, report.md, environment.json, pytrec_crosscheck.json) are in `artifacts/kaggle/<timestamp>/`; curated copies are in `results/`.

## Limits

- BEIR SciFact "test" qrels are the SciFact development claims; official SciFact test labels are not public, so results are not comparable to papers reporting the hidden test split.
- Qrels are incomplete: unjudged documents may be relevant, which bounds achievable recall and can distort per-query deltas.
- Latency is measured on a single Tesla T4 with warm-up excluded for the first query only; it is not a production serving benchmark.
- The cross-encoder reranker is a general MS MARCO model; it reduced quality at every tested depth on both dev and test, consistent with (but not proof of) domain shift on scientific claims.
- The figure (`image.png`) is rendered from `scripts/figure-data.json`, which is curated from the Kaggle `metrics.json`; regenerate with `uv run --with matplotlib python scripts/render_figure.py`.
