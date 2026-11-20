"""Tests for SciFact TSV parsing, dataset validation, and dev split."""

from __future__ import annotations

import pytest

from paper_search.data.scifact import (
    EXPECTED_CORPUS_DOCS,
    EXPECTED_QUERIES,
    EXPECTED_TEST_QUERIES,
    EXPECTED_TRAIN_QUERIES,
    build_scifact_dataset,
    parse_qrels_tsv,
)
from paper_search.data.splits import make_dev_split


def _corpus_rows(n: int) -> list[dict[str, str]]:
    return [
        {"_id": f"d{i}", "title": f"Paper {i}", "text": f"abstract text {i}"}
        for i in range(n)
    ]


def _query_rows(n: int) -> list[dict[str, str]]:
    return [
        {"_id": f"q{i}", "title": f"Claim {i}", "text": f"claim text {i}"}
        for i in range(n)
    ]


def _qrels(qids: list[str], dids: list[str]) -> dict[str, dict[str, int]]:
    return {qid: {dids[i % len(dids)]: 1} for i, qid in enumerate(qids)}


def _valid_inputs() -> tuple[
    list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, int]], dict[str, dict[str, int]]
]:
    dids = [f"d{i}" for i in range(EXPECTED_CORPUS_DOCS)]
    train_qids = [f"q{i}" for i in range(EXPECTED_TRAIN_QUERIES)]
    test_qids = [
        f"q{i}"
        for i in range(EXPECTED_TRAIN_QUERIES, EXPECTED_TRAIN_QUERIES + EXPECTED_TEST_QUERIES)
    ]
    train_qrels = _qrels(train_qids, dids)
    test_qrels = _qrels(test_qids, dids)
    query_rows = _query_rows(EXPECTED_QUERIES)
    return _corpus_rows(EXPECTED_CORPUS_DOCS), query_rows, train_qrels, test_qrels


def test_parse_qrels_tsv() -> None:
    text = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t2\nq2\td3\t1\n"
    qrels = parse_qrels_tsv(text)
    assert qrels == {"q1": {"d1": 1, "d2": 2}, "q2": {"d3": 1}}


def test_parse_qrels_tsv_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="Non-integer qrels score"):
        parse_qrels_tsv("query-id\tcorpus-id\tscore\nq1\td1\tmaybe\n")
    with pytest.raises(ValueError, match="Malformed qrels row"):
        parse_qrels_tsv("q1\td1\n")


def test_build_scifact_dataset_happy_path() -> None:
    corpus_rows, query_rows, train_qrels, test_qrels = _valid_inputs()
    data = build_scifact_dataset(corpus_rows, query_rows, train_qrels, test_qrels)
    assert data.name == "scifact"
    assert len(data.documents) == EXPECTED_CORPUS_DOCS
    assert len(data.queries) == EXPECTED_QUERIES
    assert len(data.qrels) == EXPECTED_QUERIES
    splits = {q.split for q in data.queries}
    assert splits == {"train", "test"}
    assert sum(1 for q in data.queries if q.split == "train") == EXPECTED_TRAIN_QUERIES
    assert sum(1 for q in data.queries if q.split == "test") == EXPECTED_TEST_QUERIES
    # Original IDs preserved.
    assert data.documents[0].doc_id == "d0"
    assert data.queries[0].query_id == "q0"


def test_build_scifact_dataset_rejects_wrong_counts() -> None:
    corpus_rows, query_rows, train_qrels, test_qrels = _valid_inputs()
    with pytest.raises(ValueError, match="corpus documents"):
        build_scifact_dataset(corpus_rows[:100], query_rows, train_qrels, test_qrels)


def test_build_scifact_dataset_rejects_duplicate_ids() -> None:
    corpus_rows, query_rows, train_qrels, test_qrels = _valid_inputs()
    corpus_rows[1]["_id"] = corpus_rows[0]["_id"]
    with pytest.raises(ValueError, match="duplicate document IDs"):
        build_scifact_dataset(corpus_rows, query_rows, train_qrels, test_qrels)


def test_build_scifact_dataset_rejects_unknown_qrel_refs() -> None:
    corpus_rows, query_rows, train_qrels, test_qrels = _valid_inputs()
    train_qrels["q0"]["d999999"] = 1
    with pytest.raises(ValueError, match="unknown document"):
        build_scifact_dataset(corpus_rows, query_rows, train_qrels, test_qrels)


def test_make_dev_split_deterministic() -> None:
    qids = [f"q{i:04d}" for i in range(200)]
    a = make_dev_split(qids, fraction=0.2)
    b = make_dev_split(qids, fraction=0.2)
    c = make_dev_split(list(reversed(qids)), fraction=0.2)
    assert a == b == c
    assert len(a) == 40
    assert a <= set(qids)


def test_make_dev_split_validation() -> None:
    with pytest.raises(ValueError, match="fraction"):
        make_dev_split(["q1"], fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        make_dev_split(["q1"], fraction=1.5)
    with pytest.raises(ValueError, match="empty"):
        make_dev_split([], fraction=0.2)
    with pytest.raises(ValueError, match="unique"):
        make_dev_split(["q1", "q1"], fraction=0.5)
