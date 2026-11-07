"""Hashed TF-IDF feature retriever with exact cosine similarity."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from paper_search.data.schema import Document, SearchResult, normalize_text


class DenseRetriever:
    """Dense vector retriever using L2-normalized hashed lexical representations.

    Parameters
    ----------
    embedding_dim : int, default=128
        Dimensionality of the semantic projection vector.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.embedding_dim = embedding_dim
        self.documents: list[Document] = []
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.is_indexed: bool = False

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a string into an L2-normalized hashed TF-IDF vector."""
        tokens = normalize_text(text)
        vec = np.zeros(self.embedding_dim, dtype=np.float64)

        if not tokens or not self.vocabulary or self.idf is None:
            return vec

        counts = Counter(tokens)
        for token, count in counts.items():
            if token in self.vocabulary:
                vocab_idx = self.vocabulary[token]
                dim_idx = vocab_idx % self.embedding_dim
                weight = count * self.idf[vocab_idx]
                # Alternate sign hashing to distribute term weights evenly
                sign = 1.0 if (vocab_idx % 2 == 0) else -1.0
                vec[dim_idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def index(self, documents: list[Document]) -> DenseRetriever:
        """Build vocabulary, calculate IDF, and precompute document vectors."""
        self.documents = list(documents)
        self.doc_ids = [doc.doc_id for doc in documents]
        n_docs = len(documents)

        # Build vocabulary across documents
        doc_tokens = [normalize_text(doc.full_text) for doc in documents]
        term_dfs: Counter[str] = Counter()
        for tokens in doc_tokens:
            term_dfs.update(set(tokens))

        sorted_terms = sorted(term_dfs.keys())
        self.vocabulary = {t: i for i, t in enumerate(sorted_terms)}
        n_terms = len(sorted_terms)

        idf_arr = np.zeros(n_terms, dtype=np.float64)
        for term, i in self.vocabulary.items():
            df = term_dfs[term]
            idf_arr[i] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self.idf = idf_arr

        # Encode all documents
        embeddings = np.zeros((n_docs, self.embedding_dim), dtype=np.float64)
        for i, doc in enumerate(documents):
            embeddings[i] = self._encode_text(doc.full_text)

        self.doc_embeddings = embeddings
        self.is_indexed = True
        return self

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Perform exact cosine similarity retrieval over document embeddings."""
        if k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed or self.doc_embeddings is None:
            raise RuntimeError("Retriever is not indexed yet. Call 'index' first.")

        q_vec = self._encode_text(query)
        if np.all(q_vec == 0.0):
            return []

        # Cosine similarity between query and all documents
        scores = np.dot(self.doc_embeddings, q_vec)

        # Sort descending by score, deterministic tie-breaking by doc_id
        doc_indices = np.arange(len(self.doc_ids))
        ranked_indices = sorted(
            doc_indices,
            key=lambda idx: (-scores[idx], self.doc_ids[idx]),
        )[:k]

        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                score=float(scores[idx]),
                rank=r + 1,
            )
            for r, idx in enumerate(ranked_indices)
        ]
