"""Command-line interface for research paper semantic search."""

from __future__ import annotations

from pathlib import Path

import typer

from paper_search import __version__
from paper_search.data.fixtures import load_fixture_benchmark
from paper_search.data.schema import BenchmarkDataset, Query, SearchResult
from paper_search.data.scifact import load_scifact_benchmark
from paper_search.evaluation.bootstrap import (
    PairedComparisonResult,
    compare_systems_bootstrap,
)
from paper_search.evaluation.metrics import IREvaluator, IRMetrics
from paper_search.reporting.tables import (
    format_ir_benchmark_table,
    generate_paper_search_report,
)
from paper_search.reranking.reranker import LexicalOverlapReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import HashedTfidfRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever

app = typer.Typer(help="Research Paper Semantic Search CLI.", no_args_is_help=True)

_DATASET_HELP = "Benchmark corpus: 'fixture' (offline demo) or 'scifact' (BEIR SciFact)."


def _load_dataset(name: str) -> BenchmarkDataset:
    if name == "fixture":
        return load_fixture_benchmark()
    if name == "scifact":
        return load_scifact_benchmark()
    raise typer.BadParameter(f"Unknown dataset '{name}'. Choose fixture or scifact.")


def _test_split(
    dataset: BenchmarkDataset,
) -> tuple[list[Query], dict[str, dict[str, int]]]:
    """Return test-split queries and their qrels for benchmark commands."""
    queries = [
        q for q in dataset.queries if q.split == "test" and q.query_id in dataset.qrels
    ]
    qrels = {q.query_id: dataset.qrels[q.query_id] for q in queries}
    return queries, qrels


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query or scientific claim."),  # noqa: B008
    system: str = typer.Option("hybrid", help="Retriever: 'bm25', 'tfidf', 'hybrid', or 'rerank'."),  # noqa: B008
    top_k: int = typer.Option(5, help="Number of results to return."),  # noqa: B008
    dataset: str = typer.Option("fixture", "--dataset", help=_DATASET_HELP),  # noqa: B008
) -> None:
    """Search the scientific paper corpus."""
    data = _load_dataset(dataset)
    docs = data.documents
    doc_map = {d.doc_id: d for d in docs}

    results: list[SearchResult]
    if system == "bm25":
        results = BM25Retriever().index(docs).search(query, k=top_k)
    elif system == "tfidf":
        results = HashedTfidfRetriever().index(docs).search(query, k=top_k)
    elif system == "hybrid":
        results = HybridRRFRetriever().index(docs).search(query, k=top_k)
    elif system == "rerank":
        hybrid = HybridRRFRetriever().index(docs)
        candidates = hybrid.search(query, k=20)
        reranker = LexicalOverlapReranker()
        reranker.index_documents(docs)
        results, _ = reranker.rerank(query, candidates, top_k=top_k)
    else:
        typer.echo(f"Unknown system '{system}'. Choose bm25, tfidf, hybrid, or rerank.")
        raise typer.Exit(code=1)

    typer.echo(f"\nTop {len(results)} results for '{query}' using [{system}]:\n")
    for r in results:
        doc = doc_map.get(r.doc_id)
        title = doc.title if doc else "Unknown"
        typer.echo(f"[{r.rank}] ({r.score:.4f}) {r.doc_id}: {title}")


def _run_fixture_comparison(
    data: BenchmarkDataset,
) -> tuple[dict[str, IRMetrics], PairedComparisonResult]:
    """Evaluate the CPU-safe systems on the dataset's test split."""
    queries, qrels = _test_split(data)
    docs = data.documents

    bm25 = BM25Retriever().index(docs)
    tfidf = HashedTfidfRetriever().index(docs)
    hybrid = HybridRRFRetriever().index(docs)
    reranker = LexicalOverlapReranker()
    reranker.index_documents(docs)

    runs: dict[str, dict[str, list[SearchResult]]] = {
        "BM25 Lexical": {},
        "Hashed TF-IDF": {},
        "Hybrid RRF": {},
        "Lexical Rerank": {},
    }

    for q in queries:
        bm25_res = bm25.search(q.text, k=100)
        tfidf_res = tfidf.search(q.text, k=100)
        hybrid_res = hybrid.search(q.text, k=100)
        rerank_res, _ = reranker.rerank(q.text, hybrid_res, top_k=100)

        runs["BM25 Lexical"][q.query_id] = bm25_res
        runs["Hashed TF-IDF"][q.query_id] = tfidf_res
        runs["Hybrid RRF"][q.query_id] = hybrid_res
        runs["Lexical Rerank"][q.query_id] = rerank_res

    metrics_by_system = {
        name: IREvaluator.evaluate_run(system_runs, qrels)
        for name, system_runs in runs.items()
    }
    paired_comp = compare_systems_bootstrap(
        runs["BM25 Lexical"], runs["Lexical Rerank"], qrels
    )
    return metrics_by_system, paired_comp


@app.command()
def compare(
    dataset: str = typer.Option("fixture", "--dataset", help=_DATASET_HELP),  # noqa: B008
) -> None:
    """Run benchmark comparison across BM25, hashed TF-IDF, Hybrid RRF, and rerank."""
    data = _load_dataset(dataset)
    metrics_by_system, _ = _run_fixture_comparison(data)

    if data.name == "scifact":
        typer.echo("\nSciFact results (BEIR test split; neural systems run via Kaggle):")
    else:
        typer.echo("\nSynthetic fixture results (not official SciFact):")
    typer.echo(format_ir_benchmark_table(metrics_by_system))


@app.command()
def report(
    output_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Optional path to write markdown report to.",
    ),
    dataset: str = typer.Option("fixture", "--dataset", help=_DATASET_HELP),  # noqa: B008
) -> None:
    """Generate and display complete research report."""
    data = _load_dataset(dataset)
    metrics_by_system, paired_comp = _run_fixture_comparison(data)

    doc_text = generate_paper_search_report(
        metrics_by_system, paired_comp, dataset_name=data.name
    )

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
