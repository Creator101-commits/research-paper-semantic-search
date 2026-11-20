"""BEIR SciFact loader from the Hugging Face Hub.

Downloads the pinned corpus, queries, and qrels revisions, validates record
counts and foreign keys, and preserves original benchmark IDs and splits.
Requires the ``neural`` optional extra (``huggingface_hub``, ``pyarrow``).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from paper_search.data.schema import BenchmarkDataset, Document, Query

SCIFACT_DATASET = "BeIR/scifact"
SCIFACT_QRELS_DATASET = "BeIR/scifact-qrels"
SCIFACT_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"

EXPECTED_CORPUS_DOCS = 5183
EXPECTED_QUERIES = 1109
EXPECTED_TRAIN_QUERIES = 809
EXPECTED_TEST_QUERIES = 300

_CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
_QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
_QRELS_FILES = {"train": "train.tsv", "test": "test.tsv"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_download(repo_id: str, filename: str, revision: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Loading SciFact requires the neural extra: install with "
            "`uv sync --extra neural` or `pip install .[neural]`."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
        )
    )


def _read_parquet_records(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def parse_qrels_tsv(text: str) -> dict[str, dict[str, int]]:
    """Parse a BEIR qrels TSV (`query-id`, `corpus-id`, `score`) into nested dicts."""
    qrels: dict[str, dict[str, int]] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Malformed qrels row at line {line_no}: {line!r}")
        query_id, doc_id, score_raw = parts
        if query_id == "query-id":
            continue  # header row
        try:
            score = int(score_raw)
        except ValueError as exc:
            raise ValueError(
                f"Non-integer qrels score at line {line_no}: {score_raw!r}"
            ) from exc
        qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def build_scifact_dataset(
    corpus_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    train_qrels: dict[str, dict[str, int]],
    test_qrels: dict[str, dict[str, int]],
) -> BenchmarkDataset:
    """Build and validate a BenchmarkDataset from raw BEIR SciFact records."""
    documents = [
        Document(
            doc_id=str(row["_id"]),
            title=str(row.get("title") or ""),
            abstract=str(row["text"]),
        )
        for row in corpus_rows
    ]
    doc_ids = [doc.doc_id for doc in documents]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("SciFact corpus contains duplicate document IDs.")
    if len(documents) != EXPECTED_CORPUS_DOCS:
        raise ValueError(
            f"Expected {EXPECTED_CORPUS_DOCS} SciFact corpus documents, "
            f"found {len(documents)}."
        )
    if any(not doc.abstract.strip() for doc in documents):
        raise ValueError("SciFact corpus contains a document with empty text.")

    split_by_query: dict[str, str] = {}
    for split, split_qrels in (("train", train_qrels), ("test", test_qrels)):
        for query_id in split_qrels:
            if query_id in split_by_query:
                raise ValueError(f"Query {query_id} appears in both qrels splits.")
            split_by_query[query_id] = split

    queries: list[Query] = []
    for row in query_rows:
        query_id = str(row["_id"])
        text = str(row.get("text") or "") or str(row.get("title") or "")
        query_split = split_by_query.get(query_id)
        if query_split is None:
            raise ValueError(f"Query {query_id} has no qrels in train or test split.")
        if not text.strip():
            raise ValueError(f"Query {query_id} has empty text.")
        queries.append(Query(query_id=query_id, text=text, split=query_split))

    query_ids = [query.query_id for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("SciFact queries contain duplicate query IDs.")
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_QUERIES} SciFact queries, found {len(queries)}."
        )

    n_train = sum(1 for q in queries if q.split == "train")
    n_test = sum(1 for q in queries if q.split == "test")
    if n_train != EXPECTED_TRAIN_QUERIES or n_test != EXPECTED_TEST_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_QUERIES} train and {EXPECTED_TEST_QUERIES} "
            f"test queries, found {n_train} train and {n_test} test."
        )
    if len(train_qrels) != EXPECTED_TRAIN_QUERIES or len(test_qrels) != EXPECTED_TEST_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_QUERIES} train and {EXPECTED_TEST_QUERIES} "
            f"judged test queries, found {len(train_qrels)} and {len(test_qrels)}."
        )

    doc_id_set = set(doc_ids)
    query_id_set = set(query_ids)
    for split, split_qrels in (("train", train_qrels), ("test", test_qrels)):
        for query_id, rels in split_qrels.items():
            if query_id not in query_id_set:
                raise ValueError(f"Qrels reference unknown query {query_id} ({split}).")
            for doc_id, score in rels.items():
                if doc_id not in doc_id_set:
                    raise ValueError(f"Qrels reference unknown document {doc_id} ({split}).")
                if score <= 0:
                    raise ValueError(f"Non-positive qrels score for {query_id}/{doc_id}.")

    qrels = {**train_qrels, **test_qrels}
    return BenchmarkDataset(
        name="scifact",
        documents=documents,
        queries=queries,
        qrels=qrels,
        provenance={
            "corpus_dataset": SCIFACT_DATASET,
            "corpus_revision": SCIFACT_REVISION,
            "qrels_dataset": SCIFACT_QRELS_DATASET,
            "qrels_revision": SCIFACT_QRELS_REVISION,
            "license": "cc-by-sa-4.0",
        },
    )


def load_scifact_benchmark() -> BenchmarkDataset:
    """Download and validate the pinned BEIR SciFact benchmark from the HF Hub."""
    corpus_path = _hf_download(SCIFACT_DATASET, _CORPUS_FILE, SCIFACT_REVISION)
    queries_path = _hf_download(SCIFACT_DATASET, _QUERIES_FILE, SCIFACT_REVISION)
    qrels_paths = {
        split: _hf_download(SCIFACT_QRELS_DATASET, filename, SCIFACT_QRELS_REVISION)
        for split, filename in _QRELS_FILES.items()
    }

    corpus_rows = _read_parquet_records(corpus_path)
    query_rows = _read_parquet_records(queries_path)
    split_qrels = {
        split: parse_qrels_tsv(path.read_text(encoding="utf-8"))
        for split, path in qrels_paths.items()
    }

    dataset = build_scifact_dataset(
        corpus_rows,
        query_rows,
        split_qrels["train"],
        split_qrels["test"],
    )
    provenance = {
        **dataset.provenance,
        "files": {
            f"{SCIFACT_DATASET}/{_CORPUS_FILE}": _sha256_file(corpus_path),
            f"{SCIFACT_DATASET}/{_QUERIES_FILE}": _sha256_file(queries_path),
            **{
                f"{SCIFACT_QRELS_DATASET}/{filename}": _sha256_file(path)
                for filename, path in qrels_paths.items()
            },
        },
        "counts": {
            "corpus_documents": len(dataset.documents),
            "queries": len(dataset.queries),
            "train_queries": EXPECTED_TRAIN_QUERIES,
            "test_queries": EXPECTED_TEST_QUERIES,
            "qrels": len(dataset.qrels),
        },
        "note": (
            "BEIR SciFact 'test' qrels correspond to the SciFact development claims; "
            "official SciFact test labels are not public."
        ),
    }
    return replace(dataset, provenance=provenance)
