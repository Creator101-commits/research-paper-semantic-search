> This document is a roadmap. See README.md for implemented behavior and verification limits; planned capabilities are not measured results.

# Build Plan: Research-Paper Semantic Search

## Goal

Build a reproducible information-retrieval benchmark for finding scientific papers relevant to a research claim or question. The project will compare lexical retrieval, general-purpose and scientific transformer embeddings, hybrid rank fusion, and cross-encoder reranking while measuring ranking quality, latency, index cost, and query-level failures on fixed relevance judgments.

## Scope

### Included in v1

- Acquire and validate the BEIR SciFact corpus, queries, and relevance judgments.
- Normalize titles and abstracts while preserving benchmark document and query IDs.
- Implement a deterministic BM25 lexical baseline.
- Implement dense bi-encoder retrieval with at least one compact general encoder.
- Evaluate one scientific-domain embedding model when its model terms and hardware requirements are acceptable.
- Build exact vector search first and add FAISS indexing when it preserves verified ranking behavior.
- Implement reciprocal-rank fusion for hybrid lexical and dense retrieval.
- Implement cross-encoder reranking over a configurable top-N candidate set.
- Tune fusion and reranking settings using development queries only.
- Evaluate with NDCG, MAP, Recall, Precision, and MRR at fixed cutoffs.
- Measure query latency, corpus encoding time, index construction time, index size, and memory when practical.
- Report uncertainty and query-level win/loss analysis.
- Track all runs, model revisions, data hashes, settings, metrics, and artifacts with MLflow.
- Provide a CLI and a complete research report.

### Explicitly outside v1

- Generated answers, summaries, or conversational RAG behavior.
- Crawling arbitrary copyrighted research-paper PDFs.
- Production-scale distributed retrieval.
- User accounts, search personalization, online learning, or click models.
- Training a transformer from scratch.
- Fine-tuning on final test relevance judgments.
- Changing benchmark qrels based on model outputs.
- Claiming that unjudged documents are definitively irrelevant.
- A polished web interface.

## Research questions

1. How do lexical and dense retrieval differ on scientific claims containing exact terminology versus semantic paraphrases?
2. Does a scientific-domain encoder consistently outperform a compact general-purpose encoder?
3. Does hybrid fusion improve recall without unacceptable latency or index cost?
4. How much ranking quality does cross-encoder reranking add at candidate depths of 20, 50, and 100?
5. Which query categories remain difficult for every approach?

## Stack

- **Language:** Python 3.12
- **Packaging:** `uv`, `pyproject.toml`, and locked dependencies
- **Data:** pandas, NumPy, SciPy, PyArrow, JSONL, and Parquet
- **Lexical retrieval:** BM25S
- **Dense retrieval and reranking:** sentence-transformers and PyTorch
- **Vector search:** exact NumPy search for fixtures and FAISS CPU for the benchmark index
- **Metrics:** ranx or verified internal metric functions cross-checked against a reference
- **Tracking:** MLflow
- **Visualization:** Matplotlib and seaborn
- **CLI and quality:** Typer, pytest, Ruff, mypy, and GitHub Actions

## Architecture

```text
SciFact corpus + queries + qrels
                |
                v
      validation + normalization
                |
        +-------+--------+
        |                |
        v                v
   BM25 index       embedding cache
        |                |
        v                v
 lexical ranking    vector ranking
        |                |
        +-------+--------+
                |
          optional RRF fusion
                |
                v
      optional cross-encoder reranker
                |
                v
       shared evaluator + profiler
                |
                v
       MLflow + tables + error report
```

### Components

#### `src/paper_search/data/`

Downloads the benchmark, verifies source integrity, parses corpus/query/qrel records, validates foreign keys, normalizes text, preserves benchmark splits, and writes typed tables plus a dataset manifest.

#### `src/paper_search/indexing/`

Creates versioned BM25 indexes and dense embedding indexes. Each index manifest records the corpus hash, preprocessing version, model identifier and revision, dimensionality, normalization behavior, library versions, and build time.

#### `src/paper_search/retrievers/`

Defines a shared `search(query, k)` protocol. Implementations produce ranked `(document_id, score)` results for BM25, dense retrieval, and hybrid reciprocal-rank fusion.

#### `src/paper_search/reranking/`

Scores query-document candidate pairs with a cross-encoder and returns a stable reranked list. It preserves candidate provenance and resolves tied scores deterministically.

#### `src/paper_search/evaluation/`

Computes ranking metrics from run files and qrels, records stage-level latency, performs paired query bootstrap comparisons, and produces slice results. The evaluator accepts standard query/document IDs and must not depend on a particular model.

#### `src/paper_search/reporting/`

Produces overall result tables, confidence intervals, quality-versus-latency plots, candidate-depth curves, per-query deltas, and annotated failure cases.

#### `src/paper_search/cli.py`

Exposes thin commands for download, preparation, indexing, retrieval, evaluation, comparison, and report generation.

## Data

### Primary benchmark

Use **BEIR SciFact**, a compact scientific-document retrieval benchmark containing scientific claims as queries, research-paper titles and abstracts as the corpus, and relevance judgments connecting queries to evidence-bearing documents. Its manageable corpus size allows exact retrieval checks and detailed query-level analysis on a normal computer.

### Logical schema

- `documents`: `document_id`, `title`, `abstract_text`, optional metadata
- `queries`: `query_id`, `query_text`, `split`
- `qrels`: `query_id`, `document_id`, `relevance`, `split`
- `runs`: `run_id`, `query_id`, `document_id`, `rank`, `score`, `retriever`, `latency_ms`

### Acquisition and validation

- Download through a documented BEIR-compatible source or the original benchmark release.
- Record source URL, acquisition date, archive size, checksum, benchmark version, and applicable terms.
- Verify that every qrel query and document ID exists in the corresponding table.
- Detect duplicate IDs, empty texts, invalid relevance labels, and missing published split information.
- Preserve raw files unchanged under `data/raw/scifact/` and keep them outside Git.

### Text preparation

- Preserve original text in processed tables.
- Construct one documented search field by joining title and abstract with explicit markers.
- Apply conservative Unicode and whitespace normalization only.
- Do not stem or remove stop words for dense retrieval.
- Treat BM25 tokenization as a versioned configuration and report its choices.
- Hash normalized text so stale lexical indexes and embedding caches can be rejected.

### Development and test protocol

- Preserve official benchmark partitions when both development and test judgments are available.
- If an official development set is unavailable through the selected distribution, create and version one deterministic partition from non-test judged queries before model comparison.
- Never tune fusion weights, encoder choice, reranking depth, or cutoff behavior using final test qrels.
- Keep all unjudged corpus documents eligible for retrieval while explaining incomplete judgment limitations.

## Retrieval methods

### Baselines

- Random ranking used only as an evaluator sanity check
- Title-only BM25 diagnostic baseline
- Title-plus-abstract BM25 primary lexical baseline

### Dense retrieval

- Compact general-purpose sentence-transformer bi-encoder
- Scientific-domain bi-encoder, such as a SPECTER-family model, if license and environment checks pass
- Precompute document embeddings; encode queries at runtime
- Use cosine similarity with explicitly normalized vectors unless the model documentation requires another similarity

### Hybrid retrieval

- Fuse BM25 and dense ranks using reciprocal-rank fusion.
- Tune the RRF constant or weights only on development data.
- Preserve the contribution and original rank from each source for error analysis.

### Reranking

- Retrieve candidate sets at depths 20, 50, and 100.
- Score query-title-abstract pairs with a cross-encoder.
- Measure incremental quality and latency at each depth.
- Never rerank documents that were not present in the declared candidate set.

## Evaluation design

### Primary metrics

- NDCG@10
- MAP@100
- Recall@100

### Secondary metrics

- MRR@10
- Precision@10
- Recall@10 and Recall@50
- Queries with no relevant document retrieved
- Mean, median, P95, and P99 query latency after warm-up
- Corpus encoding time, index build time, serialized index size, and peak memory when practical

### Statistical comparison

- Save query-level metrics for every run.
- Use paired query bootstrap intervals for differences between systems.
- Report counts of queries improved, harmed, or unchanged by hybrid retrieval and reranking.
- Avoid declaring superiority when the benchmark is too small to separate systems reliably.

### Error analysis

Categorize a fixed sample of failures using rules established before viewing model identities:

- exact scientific term mismatch
- acronym or abbreviation mismatch
- semantic paraphrase
- negation or contradiction
- numeric or entity specificity
- query requiring several evidence concepts
- relevant document absent from the candidate set
- candidate retrieved but incorrectly reranked

## Proposed file structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── data/scifact.yaml
│   ├── retrieval/bm25.yaml
│   ├── retrieval/dense.yaml
│   ├── retrieval/hybrid.yaml
│   └── retrieval/reranker.yaml
├── data/
│   ├── raw/                 # ignored benchmark download
│   ├── processed/           # ignored normalized tables
│   └── manifests/
├── indexes/                      # ignored BM25/vector indexes
├── src/paper_search/
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── download.py
│   │   ├── normalize.py
│   │   └── qrels.py
│   ├── indexing/
│   │   ├── bm25.py
│   │   ├── embeddings.py
│   │   └── vector_index.py
│   ├── retrievers/
│   │   ├── base.py
│   │   ├── lexical.py
│   │   ├── dense.py
│   │   └── hybrid.py
│   ├── reranking/cross_encoder.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── latency.py
│   │   └── bootstrap.py
│   └── reporting/
│       ├── tables.py
│       └── plots.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── artifacts/                    # ignored run files
├── docs/data-card.md
├── docs/research-report.md
├── pyproject.toml
└── README.md
```

## Build phases

### Phase 1: Project foundation (Status: Completed)

- **Work:** Create the package, locked environment, configuration loader, CLI, tests, linting, typing, and CI.
- **Verify:** Run a clean installation, CLI help, test suite, Ruff, mypy, and package build.
- **Complete when:** CI passes and every documented setup command works.

### Phase 2: SciFact acquisition and validation (Status: Completed)

- **Work:** Download SciFact, capture hashes and terms, parse corpus/queries/qrels, preserve IDs and splits, normalize text conservatively, and write manifests.
- **Verify:** Test unique IDs, qrel foreign keys, record counts, split isolation, and deterministic processed hashes.
- **Complete when:** Every judged query and document is valid and traceable to the source benchmark.

### Phase 3: Evaluator and BM25 baseline (Status: Completed)

- **Work:** Implement standard run files, NDCG/MAP/Recall/MRR metrics, timing, BM25 indexing, and lexical retrieval.
- **Verify:** Cross-check metrics with a reference evaluator and test ranking/tie behavior on hand-built fixtures.
- **Complete when:** The first reproducible BM25 result and latency report can be generated from configuration.

### Phase 4: Dense retrieval (Status: Completed)

- **Work:** Add batched document/query embeddings, cache manifests, exact cosine search, FAISS indexing, and one compact general encoder plus one scientific encoder.
- **Verify:** Test cache invalidation, vector dimensions, exact-versus-FAISS ranking parity, and pinned model revisions.
- **Complete when:** Both dense systems produce complete benchmark run files and resource measurements.

### Phase 5: Hybrid retrieval (Status: Completed)

- **Work:** Implement reciprocal-rank fusion with lexical/dense provenance, deterministic ties, and development-only fusion settings.
- **Verify:** Use small rankings with known fused order and confirm hybrid retrieval never drops valid source identifiers.
- **Complete when:** Hybrid quality, recall contribution, and latency can be compared with both component retrievers.

### Phase 6: Cross-encoder reranking (Status: Completed)

- **Work:** Rerank candidate depths 20, 50, and 100 in batches while preserving original ranks and document IDs.
- **Verify:** Test candidate preservation, deterministic ordering, truncation behavior, and stage-level timing.
- **Complete when:** Quality-versus-latency results identify one validation-selected reranking depth.

### Phase 7: Controlled development comparison (Status: Completed)

- **Work:** Compare lexical, dense, scientific, hybrid, and reranked systems on development queries; measure ranking quality, latency, index size, and memory.
- **Verify:** Confirm all tuning uses development qrels and every table points to saved run files.
- **Complete when:** Final test configurations and model revisions are frozen.

### Phase 8: Frozen evaluation and error analysis (Status: Completed)

- **Work:** Evaluate once on test qrels, calculate paired query-bootstrap intervals, and categorize predefined failures such as paraphrase, negation, and candidate-recall errors.
- **Verify:** Reproduce final tables from immutable runs and confirm no configuration changed after viewing test results.
- **Complete when:** Final conclusions include uncertainty, query-level wins/losses, and benchmark limitations.

### Phase 9: Documentation and release (Status: Completed)

- **Work:** Complete the data card, research report, README, architecture explanation, and reproduction commands.
- **Verify:** Run the full workflow from a clean checkout, including indexing smoke tests, evaluation, linting, typing, and CI.
- **Complete when:** Another developer can reproduce the benchmark without undocumented steps.

## Open questions

1. **Benchmark distribution:** Select the exact SciFact source and preserve its official split metadata and license information.
2. **Scientific encoder:** Choose a SPECTER-family or comparable model only after checking model terms, embedding protocol, size, and CPU feasibility.
3. **Cross-encoder:** Decide whether a general MS MARCO reranker or a scientific-domain reranker offers the clearest controlled comparison.
4. **Metric implementation:** Choose ranx or an internal evaluator after parity tests; one implementation must become authoritative for all results.
5. **Development qrels:** Confirm availability of an official development split before defining any deterministic alternative.
6. **Hybrid configuration:** Decide whether to tune a weighted fusion or use unweighted RRF as the primary method. Record the decision before final evaluation.
7. **Latency hardware:** Define the machine, thread count, batch size, warm-up, and repetition protocol used for published latency.
8. **Confidence intervals:** Use paired query bootstrap intervals, while explicitly noting that SciFact's query count limits precision.
9. **Relevance incompleteness:** Decide how unjudged documents will be displayed in qualitative examples without changing official metric labels.
