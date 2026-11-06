![Banner](./image.png)

# Research Paper Semantic Search

> A reproducible SciFact benchmark for lexical, dense, hybrid, and reranked paper retrieval.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/research-paper-semantic-search?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/research-paper-semantic-search)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/research-paper-semantic-search)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/research-paper-semantic-search)
![GitHub](https://img.shields.io/github/license/Creator101-commits/research-paper-semantic-search)

A reproducible scientific paper retrieval and ranking benchmark comparing title-plus-abstract BM25 lexical search, dense semantic encoders, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking on SciFact claims. Evaluates ranking quality (NDCG@10, MAP@100, Recall@100, MRR@10), paired query bootstrap significance, and candidate win/loss metrics.

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Evaluation and Methodology](#evaluation-and-methodology)
- [Development and Testing](#development-and-testing)
- [License](#license)

## Features

- **Okapi BM25 Lexical Retrieval**: Robertson-Sparck Jones IDF weighting with length normalization (k1=1.5, b=0.75).
- **Dense Semantic Retrieval**: High-dimensional semantic embeddings with exact cosine similarity matching.
- **Hybrid Reciprocal Rank Fusion (RRF)**: Deterministic score-agnostic rank combination with configurable smoothing constant.
- **Cross-Encoder Reranking**: Fine-grained query-document joint scoring over candidate retrieval pools.
- **Statistical Significance**: Paired per-query bootstrap difference testing for NDCG@10 with empirical confidence intervals.

## Quickstart

Run a comparison benchmark across all 4 retrieval systems in one command:

```bash
uv run paper-search compare
```

Query the corpus using hybrid retrieval:

```bash
uv run paper-search search "CRISPR Cas9 endonuclease cleavage" --system hybrid --top-k 5
```

## Installation

Ensure Python 3.12+ and `uv` are installed:

```bash
git clone https://github.com/Creator101-commits/research-paper-semantic-search.git
cd research-paper-semantic-search
uv sync
```

## CLI Reference

### 1. Version Check
```bash
uv run paper-search version
```

### 2. Search
Retrieve top results for any scientific claim or query:
```bash
uv run paper-search search "Lipid nanoparticles for mRNA delivery" --system rerank --top-k 3
```

Available systems: `bm25`, `dense`, `hybrid`, `rerank`.

### 3. Benchmark Comparison
Evaluate all systems against ground-truth relevance annotations:
```bash
uv run paper-search compare
```

### 4. Full Markdown Report
Generate an end-to-end benchmark evaluation report:
```bash
uv run paper-search report --output artifacts/scifact_benchmark_report.md
```

## Python API

```python
from paper_search.data.scifact import load_scifact_benchmark
from paper_search.retrievers.hybrid import HybridRRFRetriever
from paper_search.reranking.reranker import CrossEncoderReranker

# Load corpus
docs, queries, qrels = load_scifact_benchmark()

# Retrieve top candidates with Hybrid RRF
hybrid = HybridRRFRetriever(k_rrf=60.0).index(docs)
candidates = hybrid.search("telomere attrition in cellular aging", k=20)

# Rerank with Cross-Encoder
reranker = CrossEncoderReranker()
reranker.index_documents(docs)
final_ranked, latency_ms = reranker.rerank(
    "telomere attrition in cellular aging",
    candidates,
    top_k=5,
)

for result in final_ranked:
    print(f"[{result.rank}] Doc {result.doc_id}: score={result.score:.4f}")
```

## Evaluation and Methodology

The benchmark measures:
- **NDCG@10**: Normalized Discounted Cumulative Gain at depth 10.
- **MAP@100**: Mean Average Precision up to rank 100.
- **Recall@100 & Recall@10**: Coverage of ground-truth relevant papers.
- **MRR@10**: Mean Reciprocal Rank of the first relevant document.
- **Paired Bootstrap CI**: 95% confidence intervals on per-query NDCG delta to verify statistical superiority.

## Development and Testing

```bash
# Run test suite with branch coverage
uv run pytest

# Run linter
uv run ruff check .

# Run static type checking
uv run mypy src
```

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
