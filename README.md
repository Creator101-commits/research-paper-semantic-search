# Research Paper Semantic Search

Can lexical, dense and hybrid retrieval find the papers that actually
support a scientific claim? This project benchmarks BM25, hashed
TF-IDF, three sentence-transformer bi-encoders, weighted RRF fusion
and a cross-encoder reranker on the BEIR SciFact corpus (5,183
documents, 1,109 claims), tuning on a dev split and reading test once.

![SciFact retrieval benchmark: ranking quality per system, per-query nDCG@10 deltas vs BM25, mean-delta confidence intervals, retrieval latency, wins/ties/losses and recall@100](image.png)

The figure follows the frozen test run: (a) ranking quality per
system, (b) per-query nDCG@10 change vs BM25, (c) mean deltas with
bootstrap CIs, (d) retrieval cost, (e) wins/ties/losses, (f) recall@100.

## Results

Test split (300 queries), Tesla T4, kernel v13 @ `c66cf51`. The frozen
system is `hybrid-rrf` with no rerank.

| System | NDCG@10 | MRR@10 | Recall@100 | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: |
| **`hybrid-rrf` (frozen)** | 0.7547 | 0.7168 | 0.9693 | 29.0 |
| `dense-bge-base` | 0.7404 | 0.7034 | 0.9667 | 17.1 |
| `hybrid+ce-rerank@20` | 0.7109 | 0.6725 | 0.9693 | 150.5 |
| `hybrid+ce-rerank@50` | 0.6966 | 0.6640 | 0.9693 | 305.1 |
| `hybrid+ce-rerank@100` | 0.6902 | 0.6600 | 0.9693 | 541.0 |
| `bm25` | 0.6643 | 0.6337 | 0.8826 | 11.9 |
| `dense-minilm` | 0.6451 | 0.6047 | 0.9250 | 11.5 |
| `hybrid+lexical-rerank` | 0.5165 | 0.4770 | 0.9693 | 30.6 |
| `dense-specter` | 0.4788 | 0.4330 | 0.8399 | 16.7 |
| `tfidf` | 0.1770 | 0.1517 | 0.5083 | 1.2 |

- **The frozen hybrid wins on quality.** NDCG@10 0.7547 vs BM25
  0.6643, a paired delta of +0.0903 [+0.0605, +0.1218] with 102 wins,
  167 ties and 31 losses out of 300 queries.
- **Cross-encoder reranking reduced quality at every depth** while
  multiplying latency: vs `hybrid-rrf`, deltas are -0.0438, -0.0581
  and -0.0645 (all CIs exclude zero) at 150.5 to 541.0 ms, about
  5-19x the frozen system's 29.0 ms. Dev agreed (0.7674 down to
  0.7138/0.6962/0.6949), so `rerank_depth: null` was frozen.
- **Dense retrieval helps, but BM25 keeps a niche.** BGE-base reaches
  0.7404 (+0.0760 [+0.0401, +0.1133] vs BM25); 7 test queries are
  solved only by BM25 and 39 only by a pure dense encoder.
- **Recall@100 saturates for hybrids.** Every hybrid and reranked
  system retrieves the relevant document at 0.9693, vs 0.8826 for
  BM25, 0.9250 for MiniLM and 0.5083 for TF-IDF.

## How it works

```text
BEIR SciFact -> deterministic dev split (162 of 809 train queries)
-> tune encoder, RRF k/weights and rerank depth on dev only
-> freeze hybrid-rrf -> single test evaluation (300 queries)
-> NDCG@10/MAP/Recall/MRR/P@10 + paired bootstrap CIs vs BM25
   + pytrec_eval cross-check
```

## Run it

```bash
uv sync   # core only; --extra neural adds torch/transformers/faiss
uv run paper-search compare    # offline fixture benchmark, no downloads
uv run paper-search search "CRISPR Cas9 cleavage" --system hybrid --top-k 5
uv run pytest
uv run ruff check .
uv run mypy src
```

The full neural benchmark runs on Kaggle GPU:

```bash
./scripts/kaggle_run.sh
```

## Notes

- All tuning (encoder, fusion weights, rerank depth) happened on the
  dev split and the test split was read once; BEIR's "test" split is
  SciFact's development claims, and the official test labels are not
  public.
- Cross-encoder reranking reduced quality at every depth while
  multiplying latency (see Results); the frozen system keeps
  `rerank_depth: null`.
- Data: `BeIR/scifact` and `scifact-qrels` at pinned revisions
  (CC-BY-SA-4.0); bi-encoders and the reranker are revision-pinned;
  kernel v13 @ `c66cf51` on a Tesla T4.
- `results/scifact_benchmark.json` is the committed run output;
  `image.png` renders from it via `scripts/render_figure.py`.
