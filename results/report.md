# SciFact Retrieval Benchmark Report

- Dataset: BeIR/scifact (scifact) @ `b3b5335604bf`
- Git SHA: `c66cf51f782461ce1eac570c4a6ee8c9f812597d`
- Hardware: Tesla T4
- Frozen final system: `hybrid-rrf`
- Dev split: 162 queries tuned from 809 SciFact train queries (fraction 0.2)
- Test split: 300 BEIR 'test' queries (SciFact development claims; official test labels are not public)

## Frozen configuration (selected on dev only)

```json
{
  "dev_ndcg_at_10": 0.7673604397053564,
  "frozen_config": {
    "dense_model": "bge-base",
    "dev_fraction": 0.2,
    "dev_split_salt": "paper-search-dev-v1",
    "k_eval": 100,
    "n_bootstrap": 1000,
    "rerank_depth": null,
    "rrf": {
      "dense_weight": 2.0,
      "k": 10.0,
      "lexical_weight": 1.0
    },
    "seeds": {
      "bootstrap": 42
    }
  },
  "frozen_system": "hybrid-rrf",
  "grid_size": 60,
  "rerank_selection_dev": [
    {
      "depth": null,
      "mean_latency_ms": 28.95920085714518,
      "ndcg_at_10": 0.7673604397053564,
      "system": "hybrid-rrf"
    },
    {
      "depth": 20,
      "mean_latency_ms": 149.19320561491023,
      "ndcg_at_10": 0.7137779616863289,
      "system": "hybrid+ce-rerank@20"
    },
    {
      "depth": 50,
      "mean_latency_ms": 309.65675870186897,
      "ndcg_at_10": 0.6961907741188156,
      "system": "hybrid+ce-rerank@50"
    },
    {
      "depth": 100,
      "mean_latency_ms": 538.7115129192515,
      "ndcg_at_10": 0.6949486598446097,
      "system": "hybrid+ce-rerank@100"
    }
  ],
  "top_grid_configs": [
    {
      "dense_model": "bge-base",
      "dense_weight": 2.0,
      "k_rrf": 10.0,
      "lexical_weight": 1.0,
      "map_at_100": 0.7209083445858968,
      "ndcg_at_10": 0.7673604397053564,
      "recall_at_100": 0.9753086419753086
    },
    {
      "dense_model": "bge-base",
      "dense_weight": 2.0,
      "k_rrf": 30.0,
      "lexical_weight": 1.0,
      "map_at_100": 0.7107877799144035,
      "ndcg_at_10": 0.7578048934035919,
      "recall_at_100": 0.9814814814814815
    },
    {
      "dense_model": "bge-base",
      "dense_weight": 1.5,
      "k_rrf": 10.0,
      "lexical_weight": 1.0,
      "map_at_100": 0.7092100143177035,
      "ndcg_at_10": 0.7565395381447525,
      "recall_at_100": 0.9753086419753086
    },
    {
      "dense_model": "bge-base",
      "dense_weight": 1.5,
      "k_rrf": 30.0,
      "lexical_weight": 1.0,
      "map_at_100": 0.7044671960412701,
      "ndcg_at_10": 0.7529096292532159,
      "recall_at_100": 0.9753086419753086
    },
    {
      "dense_model": "bge-base",
      "dense_weight": 2.0,
      "k_rrf": 100.0,
      "lexical_weight": 1.0,
      "map_at_100": 0.7069063464923476,
      "ndcg_at_10": 0.7521280881876656,
      "recall_at_100": 0.9876543209876543
    }
  ]
}
```

## Test metrics

| System | NDCG@10 | MAP@100 | Recall@10 | Recall@100 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 0.6643 | 0.6266 | 0.7816 | 0.8826 | 0.6337 |
| `tfidf` | 0.1770 | 0.1578 | 0.2642 | 0.5083 | 0.1517 |
| `dense-minilm` | 0.6451 | 0.6031 | 0.7833 | 0.9250 | 0.6047 |
| `dense-bge-base` | 0.7404 | 0.6978 | 0.8742 | 0.9667 | 0.7034 |
| `dense-specter` | 0.4788 | 0.4348 | 0.6373 | 0.8399 | 0.4330 |
| `hybrid-rrf` (frozen) | 0.7547 | 0.7101 | 0.8926 | 0.9693 | 0.7168 |
| `hybrid+lexical-rerank` | 0.5165 | 0.4766 | 0.6675 | 0.9693 | 0.4770 |
| `hybrid+ce-rerank@20` | 0.7109 | 0.6645 | 0.8628 | 0.9693 | 0.6725 |
| `hybrid+ce-rerank@50` | 0.6966 | 0.6565 | 0.8289 | 0.9693 | 0.6640 |
| `hybrid+ce-rerank@100` | 0.6902 | 0.6530 | 0.8122 | 0.9693 | 0.6600 |

## Paired bootstrap vs BM25 (NDCG@10, 95% CI)

| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |
| :--- | :--- | :--- | :--- | :--- |
| `tfidf vs bm25` | -0.4873 [-0.5326, -0.4413] | 2 | 94 | 204 |
| `dense-minilm vs bm25` | -0.0193 [-0.0563, 0.0173] | 66 | 161 | 73 |
| `dense-bge-base vs bm25` | +0.0760 [0.0401, 0.1133] | 93 | 162 | 45 |
| `dense-specter vs bm25` | -0.1856 [-0.2320, -0.1337] | 47 | 123 | 130 |
| `hybrid-rrf vs bm25` | +0.0903 [0.0605, 0.1218] | 102 | 167 | 31 |
| `hybrid+lexical-rerank vs bm25` | -0.1479 [-0.1859, -0.1100] | 35 | 155 | 110 |
| `hybrid+ce-rerank@20 vs bm25` | +0.0465 [0.0152, 0.0778] | 72 | 177 | 51 |
| `hybrid+ce-rerank@50 vs bm25` | +0.0322 [-0.0007, 0.0634] | 66 | 181 | 53 |
| `hybrid+ce-rerank@100 vs bm25` | +0.0258 [-0.0086, 0.0573] | 65 | 182 | 53 |

## Additional paired comparisons (NDCG@10, 95% CI)

| Comparison | Mean Delta NDCG@10 (95% CI) | Wins | Ties | Losses |
| :--- | :--- | :--- | :--- | :--- |
| `hybrid-rrf vs dense-bge-base` | +0.0143 [+0.0003, +0.0281] | 42 | 226 | 32 |
| `hybrid+ce-rerank@20 vs hybrid-rrf` | -0.0438 [-0.0716, -0.0177] | 46 | 176 | 78 |
| `hybrid+ce-rerank@50 vs hybrid-rrf` | -0.0581 [-0.0879, -0.0302] | 44 | 175 | 81 |
| `hybrid+ce-rerank@100 vs hybrid-rrf` | -0.0645 [-0.0946, -0.0364] | 44 | 174 | 82 |

## Per-system latency (test split, end-to-end ms/query)

| System | Mean e2e | Median e2e | P95 e2e | Rerank-stage mean |
| :--- | ---: | ---: | ---: | ---: |
| `bm25` | 11.89 | 11.39 | 21.72 | - |
| `tfidf` | 1.17 | 1.12 | 1.39 | - |
| `dense-minilm` | 11.53 | 9.94 | 10.81 | - |
| `dense-bge-base` | 17.09 | 16.97 | 18.48 | - |
| `dense-specter` | 16.66 | 16.58 | 18.32 | - |
| `hybrid-rrf` | 28.98 | 28.36 | 39.76 | - |
| `hybrid+lexical-rerank` | 30.62 | 30.07 | 41.70 | 1.64 |
| `hybrid+ce-rerank@20` | 150.46 | 150.20 | 165.09 | 121.48 |
| `hybrid+ce-rerank@50` | 305.13 | 301.29 | 341.13 | 276.15 |
| `hybrid+ce-rerank@100` | 540.96 | 537.30 | 618.51 | 511.98 |

## Hard queries (NDCG@10 breakdown across test systems)

Success buckets compare pure systems only: BM25 against the pure dense encoders; hybrids and rerankers are excluded because they all consume BM25 candidates.

- Every system scores NDCG@10 = 0: 17 queries
  - `1110`: "Suboptimal nutrition is not predictive of chronic disease"
  - `1175`: "The PPR MDA5 has two N-terminal CARD domains."
  - `1199`: "The benefits of colchicine were achieved with effective widespread use of secondary prevention strategies such as high-dose statins."
  - `1213`: "The deregulated and prolonged activation of monocytes has deleterious effects in inflammatory diseases."
  - `1226`: "The loss of the TET protein functions may have dire biological consequences, such as myeloid cancers."
- BM25 scores NDCG@10 > 0 while every pure dense encoder scores 0: 7 queries
  - `1024`: "Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes."
  - `1179`: "The PRR MDA5 has a central DExD/H RNA helices domain."
  - `128`: "Arterioles have a larger lumen diameter than venules."
  - `1362`: "Venules have a larger lumen diameter than arterioles."
  - `1363`: "Venules have a thinner or absent smooth layer compared to arterioles."
- A pure dense encoder scores NDCG@10 > 0 while BM25 scores 0: 39 queries
  - `1`: "0-dimensional biomaterials show inductive properties."
  - `1049`: "Ribosomopathies have a low degree of cell and tissue specific pathology."
  - `1088`: "Silencing of Bcl2 is important for the maintenance and progression of tumors."
  - `1191`: "The amount of publicly available DNA data doubles every 10 years."
  - `1196`: "The availability of safe places to study is effective at decreasing homelessness."
- Head-to-head bm25_vs_dense-bge-base: `bm25_only` scores > 0 only: 10 queries
  - `1024`: "Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes."
  - `1179`: "The PRR MDA5 has a central DExD/H RNA helices domain."
  - `1272`: "The single flash-evoked ERG b-wave is generated by activity of ON-bipolar cells."
  - `128`: "Arterioles have a larger lumen diameter than venules."
  - `1362`: "Venules have a larger lumen diameter than arterioles."
- Head-to-head bm25_vs_dense-bge-base: `dense-bge-base_only` scores > 0 only: 35 queries
  - `1049`: "Ribosomopathies have a low degree of cell and tissue specific pathology."
  - `1088`: "Silencing of Bcl2 is important for the maintenance and progression of tumors."
  - `1191`: "The amount of publicly available DNA data doubles every 10 years."
  - `1196`: "The availability of safe places to study is effective at decreasing homelessness."
  - `1197`: "The availability of safe places to study is not effective at decreasing homelessness."

## Provenance

```json
{
  "corpus_dataset": "BeIR/scifact",
  "corpus_revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
  "counts": {
    "corpus_documents": 5183,
    "qrels": 1109,
    "queries": 1109,
    "test_queries": 300,
    "train_queries": 809
  },
  "files": {
    "BeIR/scifact-qrels/test": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    "BeIR/scifact-qrels/train": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    "BeIR/scifact/corpus/corpus-00000-of-00001.parquet": "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
    "BeIR/scifact/queries/queries-00000-of-00001.parquet": "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034"
  },
  "license": "cc-by-sa-4.0",
  "note": "BEIR SciFact 'test' qrels correspond to the SciFact development claims; official SciFact test labels are not public.",
  "qrels_dataset": "BeIR/scifact-qrels",
  "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
}
```
