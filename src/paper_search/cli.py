"""Command-line interface for research paper semantic search."""

from __future__ import annotations

from pathlib import Path

import typer

from paper_search import __version__
from paper_search.data.schema import SearchResult
from paper_search.data.scifact import load_scifact_benchmark
from paper_search.evaluation.bootstrap import compare_systems_bootstrap
from paper_search.evaluation.metrics import IREvaluator, IRMetrics
from paper_search.reporting.tables import (
    format_ir_benchmark_table,
    generate_paper_search_report,
)
from paper_search.reranking.reranker import CrossEncoderReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever

app = typer.Typer(help="Research Paper Semantic Search CLI.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query or scientific claim."),
    system: str = typer.Option("hybrid", help="Retriever: 'bm25', 'dense', 'hybrid', or 'rerank'."),  # noqa: B008
    top_k: int = typer.Option(5, help="Number of results to return."),  # noqa: B008
) -> None:
    """Search the scientific paper corpus."""
    docs, _, _ = load_scifact_benchmark()
    doc_map = {d.doc_id: d for d in docs}

    results: list[SearchResult] = []
    if system == "bm25":
        bm25_retriever = BM25Retriever().index(docs)
        results = bm25_retriever.search(query, k=top_k)
    elif system == "dense":
        dense_retriever = DenseRetriever().index(docs)
        results = dense_retriever.search(query, k=top_k)
    elif system == "hybrid":
        hybrid_retriever = HybridRRFRetriever().index(docs)
        results = hybrid_retriever.search(query, k=top_k)
    elif system == "rerank":
        hybrid = HybridRRFRetriever().index(docs)
        candidates = hybrid.search(query, k=20)
        reranker = CrossEncoderReranker()
        reranker.index_documents(docs)
        results, _ = reranker.rerank(query, candidates, top_k=top_k)
    else:
        typer.echo(f"Unknown system '{system}'. Choose bm25, dense, hybrid, or rerank.")
        raise typer.Exit(code=1)

    typer.echo(f"\nTop {len(results)} results for '{query}' using [{system}]:\n")
    for r in results:
        doc = doc_map.get(r.doc_id)
        title = doc.title if doc else "Unknown"
        typer.echo(f"[{r.rank}] ({r.score:.4f}) {r.doc_id}: {title}")


@app.command()
def compare() -> None:
    """Run benchmark comparison across BM25, Dense, Hybrid RRF, and Cross-Encoder."""
    docs, queries, qrels = load_scifact_benchmark()
    queries = [q for q in queries if q.split == "test"]
    qrels = {q.query_id: qrels[q.query_id] for q in queries}

    bm25 = BM25Retriever().index(docs)
    dense = DenseRetriever().index(docs)
    hybrid = HybridRRFRetriever().index(docs)
    reranker = CrossEncoderReranker()
    reranker.index_documents(docs)

    bm25_runs: dict[str, list[SearchResult]] = {}
    dense_runs: dict[str, list[SearchResult]] = {}
    hybrid_runs: dict[str, list[SearchResult]] = {}
    rerank_runs: dict[str, list[SearchResult]] = {}

    for q in queries:
        bm25_res = bm25.search(q.text, k=100)
        dense_res = dense.search(q.text, k=100)
        hybrid_res = hybrid.search(q.text, k=100)
        rerank_res, _ = reranker.rerank(q.text, hybrid_res, top_k=100)

        bm25_runs[q.query_id] = bm25_res
        dense_runs[q.query_id] = dense_res
        hybrid_runs[q.query_id] = hybrid_res
        rerank_runs[q.query_id] = rerank_res

    metrics_by_system: dict[str, IRMetrics] = {
        "BM25 Lexical": IREvaluator.evaluate_run(bm25_runs, qrels),
        "Hashed TF-IDF": IREvaluator.evaluate_run(dense_runs, qrels),
        "Hybrid RRF": IREvaluator.evaluate_run(hybrid_runs, qrels),
        "Lexical Rerank": IREvaluator.evaluate_run(rerank_runs, qrels),
    }

    typer.echo("\nSynthetic IR Demo Results (not official SciFact):")
    typer.echo(format_ir_benchmark_table(metrics_by_system))


@app.command()
def report(
    output_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Optional path to write markdown report to.",
    ),
) -> None:
    """Generate and display complete research report."""
    docs, queries, qrels = load_scifact_benchmark()
    queries = [q for q in queries if q.split == "test"]
    qrels = {q.query_id: qrels[q.query_id] for q in queries}

    bm25 = BM25Retriever().index(docs)
    dense = DenseRetriever().index(docs)
    hybrid = HybridRRFRetriever().index(docs)
    reranker = CrossEncoderReranker()
    reranker.index_documents(docs)

    bm25_runs: dict[str, list[SearchResult]] = {}
    dense_runs: dict[str, list[SearchResult]] = {}
    hybrid_runs: dict[str, list[SearchResult]] = {}
    rerank_runs: dict[str, list[SearchResult]] = {}

    for q in queries:
        bm25_runs[q.query_id] = bm25.search(q.text, k=100)
        dense_runs[q.query_id] = dense.search(q.text, k=100)
        hybrid_res = hybrid.search(q.text, k=100)
        hybrid_runs[q.query_id] = hybrid_res
        rerank_res, _ = reranker.rerank(q.text, hybrid_res, top_k=100)
        rerank_runs[q.query_id] = rerank_res

    metrics_by_system: dict[str, IRMetrics] = {
        "BM25 Lexical": IREvaluator.evaluate_run(bm25_runs, qrels),
        "Hashed TF-IDF": IREvaluator.evaluate_run(dense_runs, qrels),
        "Hybrid RRF": IREvaluator.evaluate_run(hybrid_runs, qrels),
        "Lexical Rerank": IREvaluator.evaluate_run(rerank_runs, qrels),
    }

    paired_comp = compare_systems_bootstrap(bm25_runs, rerank_runs, qrels)
    doc_text = generate_paper_search_report(metrics_by_system, paired_comp)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(doc_text, encoding="utf-8")
        typer.echo(f"Saved report to {output_file}")
    else:
        typer.echo(doc_text)


@app.callback()
def _main() -> None:
    """Run Paper Search commands."""


if __name__ == "__main__":
    app()
