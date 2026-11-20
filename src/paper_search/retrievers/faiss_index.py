"""Optional FAISS IndexFlatIP backend; requires the neural extra (faiss-cpu)."""

from __future__ import annotations

from typing import Any

import numpy as np


class FaissFlatIPIndex:
    """Brute-force inner-product index over L2-normalized float32 vectors.

    ``IndexFlatIP`` computes exact inner products, so cosine similarity equals
    the inner product when rows are unit-normalized. Ranking order is finalized
    by the caller, which re-sorts by score then document ID for determinism.
    """

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "The FAISS backend requires the neural extra: install with "
                "`uv sync --extra neural` or `pip install .[neural]`."
            ) from exc
        self._faiss = faiss
        self.index: Any = None
        self.dim: int = 0
        self.n_docs: int = 0

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> FaissFlatIPIndex:
        """Build an index from an (n_docs, dim) float32 matrix."""
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("FAISS index requires a nonempty 2-D embedding matrix.")
        instance = cls()
        instance.index = instance._faiss.IndexFlatIP(matrix.shape[1])
        instance.index.add(matrix)
        instance.dim = matrix.shape[1]
        instance.n_docs = matrix.shape[0]
        return instance

    @property
    def is_built(self) -> bool:
        return self.index is not None

    @property
    def size_bytes(self) -> int:
        """Vector payload size; FAISS object overhead is not included."""
        return int(self.dim) * self.n_docs * np.dtype(np.float32).itemsize

    def search(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, doc_indices) for the top k inner products."""
        if not self.is_built:
            raise RuntimeError("FAISS index is not built.")
        if k <= 0:
            raise ValueError("Result count must be positive.")
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError("Query embedding dimension does not match the index.")
        scores, indices = self.index.search(query, min(k, self.n_docs))
        return np.asarray(scores[0], dtype=np.float32), np.asarray(indices[0], dtype=np.int64)
