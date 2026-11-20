"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from paper_search.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_cli_search_bm25_default_fixture() -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "CRISPR-Cas9 gene editing in human cells",
            "--system",
            "bm25",
        ],
    )
    assert result.exit_code == 0
    assert "CRISPR-Cas9 gene editing in human cells" in result.output
    assert "doc_1" in result.output
    assert "using [bm25]" in result.output


def test_cli_search_rejects_invalid_system() -> None:
    result = runner.invoke(app, ["search", "test query", "--system", "neural"])
    assert result.exit_code != 0


def test_cli_search_rejects_invalid_dataset() -> None:
    result = runner.invoke(
        app, ["search", "test query", "--dataset", "wikipedia"]
    )
    assert result.exit_code != 0


def test_cli_search_fixture_tfidf() -> None:
    result = runner.invoke(
        app,
        ["search", "CRISPR gene editing", "--system", "tfidf"],
    )
    assert result.exit_code == 0
    assert "Hashed TF-IDF" not in result.output  # system label, not printed
    assert "using [tfidf]" in result.output
    assert "doc_1" in result.output


def test_cli_search_fixture_rerank() -> None:
    result = runner.invoke(
        app,
        ["search", "Cas9 nuclease DNA", "--system", "rerank"],
    )
    assert result.exit_code == 0
    assert "using [rerank]" in result.output
    assert "doc_1" in result.output


def test_cli_compare_fixture() -> None:
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 0
    assert "Synthetic fixture results (not official SciFact)" in result.output
    assert "BM25 Lexical" in result.output
    assert "Hashed TF-IDF" in result.output
    assert "Hybrid RRF" in result.output


def test_cli_report_fixture() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "Hand-written synthetic fixture" in result.output
    assert "not official SciFact" in result.output
