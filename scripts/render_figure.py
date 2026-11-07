"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from paper_search.data.scifact import load_scifact_benchmark
from paper_search.evaluation.bootstrap import compare_systems_bootstrap
from paper_search.evaluation.metrics import IREvaluator
from paper_search.reranking.reranker import CrossEncoderReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


docs, queries, qrels = load_scifact_benchmark()
queries = [q for q in queries if q.split == "test"]
qrels = {q.query_id: qrels[q.query_id] for q in queries}
bm25 = BM25Retriever().index(docs)
dense = DenseRetriever().index(docs)
hybrid = HybridRRFRetriever().index(docs)
reranker = CrossEncoderReranker()
reranker.index_documents(docs)
runs = {name: {} for name in ["BM25", "Hashed TF-IDF", "Hybrid RRF", "Lexical rerank"]}
for q in queries:
    runs["BM25"][q.query_id] = bm25.search(q.text, k=100)
    runs["Hashed TF-IDF"][q.query_id] = dense.search(q.text, k=100)
    candidates = hybrid.search(q.text, k=100)
    runs["Hybrid RRF"][q.query_id] = candidates
    runs["Lexical rerank"][q.query_id] = reranker.rerank(
        q.text, candidates, top_k=100, max_candidate_depth=100
    )[0]
metrics = {n: IREvaluator.evaluate_run(r, qrels) for n, r in runs.items()}
x = np.arange(len(runs))
axes[0].bar(
    x - 0.18, [m.ndcg_at_10 for m in metrics.values()], 0.36, label="NDCG@10", color=COLORS[0]
)
axes[0].bar(
    x + 0.18, [m.mrr_at_10 for m in metrics.values()], 0.36, label="MRR@10", color=COLORS[1]
)
axes[0].set(
    xticks=x, xticklabels=list(runs), ylim=(0, 1.16), title="Measured synthetic test queries"
)
axes[0].tick_params(axis="x", rotation=25)
axes[0].legend(fontsize=9)
first = runs["BM25"][queries[0].query_id][:5]
axes[1].barh([r.doc_id for r in first], [r.score for r in first], color=COLORS[0])
axes[1].invert_yaxis()
axes[1].set(xlabel="BM25 score", title="First test claim: top documents")
wins = []
ties = []
losses = []
for name in list(runs)[1:]:
    result = compare_systems_bootstrap(runs["BM25"], runs[name], qrels)
    wins.append(result.wins)
    ties.append(result.ties)
    losses.append(result.losses)
x = np.arange(3)
axes[2].bar(x, wins, color=COLORS[1], label="Wins")
axes[2].bar(x, ties, bottom=wins, color=COLORS[4], label="Ties")
axes[2].bar(x, losses, bottom=np.array(wins) + ties, color=COLORS[2], label="Losses")
axes[2].set(
    xticks=x,
    xticklabels=list(runs)[1:],
    ylabel="Number of test queries",
    title="NDCG@10 against BM25",
    ylim=(0, 6.2),
)
axes[2].tick_params(axis="x", rotation=25)
axes[2].legend(fontsize=8)
fig.suptitle("Paper retrieval | lexical demonstration on hand-written fixtures", fontsize=16)
fig.supxlabel(
    "Ten synthetic documents, five test queries. No SciFact records, neural encoder, "
    "or cross-encoder is loaded.",
    fontsize=10,
)
payload = {
    "source": "synthetic fixtures, test split only",
    "ndcg_at_10": {n: m.ndcg_at_10 for n, m in metrics.items()},
    "wins": wins,
    "ties": ties,
    "losses": losses,
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
