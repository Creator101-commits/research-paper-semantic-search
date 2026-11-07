"""Review regressions for ranking validity and fixed result counts."""

import pytest

from paper_search.data.schema import SearchResult
from paper_search.data.scifact import load_scifact_benchmark
from paper_search.evaluation.bootstrap import compare_systems_bootstrap
from paper_search.evaluation.metrics import IREvaluator
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever


def test_repeated_relevant_doc_cannot_inflate_metrics():
    ranking = [SearchResult("a", 1.0, 1), SearchResult("a", 0.5, 2)]
    with pytest.raises(ValueError, match="duplicate"):
        IREvaluator.evaluate_query(ranking, {"a": 1})


@pytest.mark.parametrize("cls", [BM25Retriever, DenseRetriever])
def test_nonpositive_search_count_rejected(cls):
    docs, _, _ = load_scifact_benchmark()
    retriever = cls().index(docs)
    with pytest.raises(ValueError):
        retriever.search("CRISPR", k=-1)


def test_empty_bootstrap_cannot_publish_nan():
    with pytest.raises(ValueError):
        compare_systems_bootstrap({}, {}, {})
