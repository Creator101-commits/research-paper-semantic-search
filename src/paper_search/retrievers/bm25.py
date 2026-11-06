"""BM25 (Robertson-Spärck Jones) lexical retriever."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from paper_search.data.schema import Document, SearchResult, normalize_text


class BM25Retriever:
    """Okapi BM25 lexical information retrieval engine.

    Parameters
    ----------
    k1 : float, default=1.5
        BM25 term-frequency saturation parameter.
    b : float, default=0.75
        BM25 document-length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.doc_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.is_indexed: bool = False

    def index(self, documents: list[Document]) -> BM25Retriever:
        """Build inverted index and compute corpus statistics."""
        self.documents = list(documents)
        self.doc_ids = [doc.doc_id for doc in documents]
        self.inverted_index.clear()
        self.idf.clear()
        self.doc_lens.clear()

        n_docs = len(documents)
        if n_docs == 0:
            raise ValueError("Cannot build index over empty document collection.")

        doc_frequencies: Counter[str] = Counter()

        for doc_idx, doc in enumerate(documents):
            tokens = normalize_text(doc.full_text)
            self.doc_lens.append(len(tokens))
            tf_counts = Counter(tokens)

            for term, count in tf_counts.items():
                self.inverted_index[term].append((doc_idx, count))
                doc_frequencies[term] += 1

        self.avg_doc_len = sum(self.doc_lens) / n_docs

        # Calculate Robertson-Spärck Jones IDF
        for term, df in doc_frequencies.items():
            idf_val = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, idf_val)

        self.is_indexed = True
        return self

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Retrieve top k documents for a query."""
        if not self.is_indexed:
            raise RuntimeError("Retriever is not indexed yet. Call 'index' first.")

        query_tokens = normalize_text(query)
        if not query_tokens:
            return []

        doc_scores: dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                d_len = self.doc_lens[doc_idx]
                norm_len = 1.0 - self.b + self.b * (d_len / self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * norm_len
                doc_scores[doc_idx] += idf * (num / denom)

        # Sort descending by score, deterministic tie-breaking by doc_id
        ranked = sorted(
            doc_scores.items(),
            key=lambda item: (-item[1], self.doc_ids[item[0]]),
        )[:k]

        return [
            SearchResult(
                doc_id=self.doc_ids[doc_idx],
                score=float(score),
                rank=r + 1,
            )
            for r, (doc_idx, score) in enumerate(ranked)
        ]
