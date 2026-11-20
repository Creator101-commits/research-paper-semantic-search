"""Sentence-transformers cross-encoder pair scorer; requires the neural extra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class SentenceTransformerPairScorer:
    """Cross-encoder scorer over (query, passage) pairs.

    Model weights are pinned to an explicit Hugging Face revision. Passages
    longer than the model context are truncated by the underlying encoder.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CROSS_ENCODER_MODEL_ID,
        *,
        revision: str | None = DEFAULT_CROSS_ENCODER_REVISION,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformerPairScorer requires the neural extra: install "
                "with `uv sync --extra neural` or `pip install .[neural]`."
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CrossEncoder(
            model_id, revision=revision, device=self.device
        )

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Return one relevance score per passage for the given query."""
        if not texts:
            return np.zeros(0, dtype=np.float64)
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float64)
