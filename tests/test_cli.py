"""Tests for Paper Search CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from paper_search.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_search_bm25() -> None:
    result = runner.invoke(app, ["search", "CRISPR Cas9 DNA", "--system", "bm25", "--top-k", "2"])
    assert result.exit_code == 0
    assert "doc_1" in result.output


def test_cli_search_dense() -> None:
    result = runner.invoke(app, ["search", "CRISPR Cas9", "--system", "dense", "--top-k", "2"])
    assert result.exit_code == 0
    assert "Top 2 results" in result.output
    assert "doc_1" in result.output


def test_cli_search_hybrid() -> None:
    args = ["search", "mRNA lipid vaccine", "--system", "hybrid", "--top-k", "2"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Top 2 results" in result.output
    assert "doc_2" in result.output


def test_cli_search_rerank() -> None:
    args = ["search", "mitochondrial oxidative stress", "--system", "rerank", "--top-k", "2"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Top 2 results" in result.output
    assert "doc_7" in result.output


def test_cli_search_invalid_system() -> None:
    result = runner.invoke(app, ["search", "query", "--system", "unsupported"])
    assert result.exit_code == 1
    assert "Unknown system" in result.output


def test_cli_compare() -> None:
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 0
    assert "Synthetic IR Demo Results" in result.output
    assert "NDCG@10" in result.output


def test_cli_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["report", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Research Paper Semantic Search Benchmark Report" in content


def test_cli_report_stdout() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "Research Paper Semantic Search Benchmark Report" in result.output
