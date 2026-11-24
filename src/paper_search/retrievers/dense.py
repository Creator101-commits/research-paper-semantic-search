"""Dense retrievers: hashed TF-IDF baseline and neural sentence encoders."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from paper_search.data.schema import Document, SearchResult, normalize_text
from paper_search.retrievers.faiss_index import FaissFlatIPIndex

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class HashedTfidfRetriever:
    """Dense retriever over L2-normalized hashed TF-IDF vectors.

    This is a lexical baseline, not a neural encoder. Vectors are hashed into
    a fixed dimension with sign alternation and scored by exact inner product.

    Parameters
    ----------
    embedding_dim : int, default=128
        Dimensionality of the hashed feature vector.
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
        self._doc_order: np.ndarray | None = None
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

    def index(self, documents: list[Document]) -> HashedTfidfRetriever:
        """Build vocabulary, calculate IDF, and precompute document vectors."""
        self.documents = list(documents)
        self.doc_ids = [doc.doc_id for doc in documents]
        n_docs = len(documents)

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

        embeddings = np.zeros((n_docs, self.embedding_dim), dtype=np.float64)
        for i, doc in enumerate(documents):
            embeddings[i] = self._encode_text(doc.full_text)

        self.doc_embeddings = embeddings
        id_order = np.argsort(np.asarray(self.doc_ids))
        self._doc_order = np.empty(len(self.doc_ids), dtype=np.int64)
        self._doc_order[id_order] = np.arange(len(self.doc_ids))
        self.is_indexed = True
        return self

    def _topk(self, scores: np.ndarray, k: int) -> list[SearchResult]:
        """Rank by score descending with document ID ascending tie breaks."""
        assert self._doc_order is not None
        order = np.lexsort((self._doc_order, -scores))[:k]
        return [
            SearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank,
            )
            for rank, idx in enumerate(order, start=1)
        ]

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
        return self._topk(scores, k)


class TextEncoder(Protocol):
    """Text embedding interface; document and query encodings may differ."""

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Sentence-transformers bi-encoder with optional query-side prompt prefix.

    Requires the ``neural`` optional extra. Model weights are pinned to an
    explicit Hugging Face revision for reproducibility.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = DEFAULT_MODEL_REVISION,
        query_prompt: str = "",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformerEncoder requires the neural extra: install with "
                "`uv sync --extra neural` or `pip install .[neural]`."
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.query_prompt = query_prompt
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(
            model_id, revision=revision, device=self.device
        )
        self._dim = int(self._model.get_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus texts without any prompt prefix."""
        return self._encode(texts, prompt="")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode queries with the configured instruction prefix."""
        return self._encode(texts, prompt=self.query_prompt)

    def _encode(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        items = [prompt + text for text in texts] if prompt else list(texts)
        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeddings = self._model.encode(
            items,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class SentenceTransformerRetriever:
    """Dense retriever over an injected text encoder with exact or FAISS search.

    Document embeddings are L2-normalized once at index time, so inner product
    equals cosine similarity. Ranking ties resolve by document ID ascending.
    """

    def __init__(
        self,
        encoder: TextEncoder | None = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = DEFAULT_MODEL_REVISION,
        query_prompt: str = "",
        backends: Sequence[str] = ("exact",),
        name: str | None = None,
    ) -> None:
        if encoder is None:
            encoder = SentenceTransformerEncoder(
                model_id=model_id,
                revision=revision,
                query_prompt=query_prompt,
            )
        self.encoder = encoder
        self.backends = tuple(backends)
        unknown = set(self.backends) - {"exact", "faiss"}
        if unknown or not self.backends:
            raise ValueError(f"Unknown or empty backends: {sorted(unknown) or self.backends}")
        self.name = name or type(encoder).__name__
        self.documents: list[Document] = []
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None
        self.embedding_dim: int = 0
        self._doc_order: np.ndarray | None = None
        self._faiss: FaissFlatIPIndex | None = None
        self.encode_seconds: float = 0.0
        self.index_seconds: float = 0.0
        self.faiss_build_seconds: float = 0.0
        self.is_indexed: bool = False

    def index(self, documents: list[Document]) -> SentenceTransformerRetriever:
        """Encode documents and build the configured search backends."""
        self.documents = list(documents)
        self.doc_ids = [doc.doc_id for doc in documents]
        if not self.documents:
            raise ValueError("Cannot build index over empty document collection.")

        t0 = time.perf_counter()
        embeddings = self.encoder.encode_documents(
            [doc.plain_text for doc in self.documents]
        )
        self.encode_seconds = time.perf_counter() - t0

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.documents):
            raise ValueError("Encoder must return one embedding per document.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=matrix, where=norms > 0)
        self.doc_embeddings = np.ascontiguousarray(matrix)
        self.embedding_dim = int(matrix.shape[1])

        # Position of each document in doc_id-ascending order for tie breaks.
        id_order = np.argsort(np.asarray(self.doc_ids))
        self._doc_order = np.empty(len(self.doc_ids), dtype=np.int64)
        self._doc_order[id_order] = np.arange(len(self.doc_ids))

        t1 = time.perf_counter()
        if "faiss" in self.backends:
            faiss_t0 = time.perf_counter()
            self._faiss = FaissFlatIPIndex.from_embeddings(self.doc_embeddings)
            self.faiss_build_seconds = time.perf_counter() - faiss_t0
        self.index_seconds = time.perf_counter() - t1
        self.is_indexed = True
        return self

    @property
    def index_size_bytes(self) -> int:
        """Bytes for stored embeddings plus the FAISS vector payload if built."""
        if self.doc_embeddings is None:
            return 0
        total = int(self.doc_embeddings.nbytes)
        if self._faiss is not None:
            total += self._faiss.size_bytes
        return total

    def encoder_info(self) -> dict[str, Any]:
        """Provenance metadata for the injected encoder, when available."""
        return {
            "encoder": type(self.encoder).__name__,
            "model_id": getattr(self.encoder, "model_id", None),
            "revision": getattr(self.encoder, "revision", None),
            "query_prompt": getattr(self.encoder, "query_prompt", ""),
            "device": getattr(self.encoder, "device", None),
            "embedding_dim": self.embedding_dim,
            "backends": list(self.backends),
            "encode_seconds": self.encode_seconds,
            "index_seconds": self.index_seconds,
            "faiss_build_seconds": self.faiss_build_seconds,
            "index_size_bytes": self.index_size_bytes,
        }

    def _encode_query(self, query: str) -> np.ndarray:
        vector = np.asarray(
            self.encoder.encode_queries([query]), dtype=np.float32
        ).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def _topk(self, scores: np.ndarray, k: int) -> list[SearchResult]:
        """Rank by score descending with document ID ascending tie breaks."""
        assert self._doc_order is not None
        order = np.lexsort((self._doc_order, -scores))[:k]
        return [
            SearchResult(
                doc_id=self.doc_ids[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank,
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def search(
        self,
        query: str,
        k: int = 10,
        backend: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve top k documents; ``backend`` overrides the default."""
        if k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed or self.doc_embeddings is None:
            raise RuntimeError("Retriever is not indexed yet. Call 'index' first.")

        backend = backend or self.backends[0]
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' was not built for this retriever.")

        q_vec = self._encode_query(query)
        if not np.any(q_vec):
            return []

        if backend == "exact":
            scores = self.doc_embeddings @ q_vec
            return self._topk(scores, k)

        assert self._faiss is not None
        assert self._doc_order is not None
        overfetch = min(len(self.doc_ids), max(k + 64, 2 * k))
        faiss_scores, faiss_indices = self._faiss.search(q_vec, overfetch)
        valid = faiss_indices >= 0
        scores = faiss_scores[valid]
        indices = faiss_indices[valid]
        # Re-rank the overfetched candidates with the shared tie-break rule.
        order = np.lexsort((self._doc_order[indices], -scores))[:k]
        return [
            SearchResult(
                doc_id=self.doc_ids[int(indices[i])],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]
