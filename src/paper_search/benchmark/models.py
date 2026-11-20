"""Pinned model registry for the neural SciFact benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from paper_search.reranking.sentence_transformer import (
    DEFAULT_CROSS_ENCODER_MODEL_ID,
    DEFAULT_CROSS_ENCODER_REVISION,
)
from paper_search.retrievers.dense import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class DenseModelSpec:
    """One bi-encoder configuration evaluated by the benchmark."""

    key: str
    model_id: str
    revision: str
    query_prompt: str = ""
    description: str = ""


DENSE_MODEL_SPECS: tuple[DenseModelSpec, ...] = (
    DenseModelSpec(
        key="minilm",
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        description="all-MiniLM-L6-v2, compact general-purpose encoder",
    ),
    DenseModelSpec(
        key="bge-base",
        model_id="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        query_prompt=BGE_QUERY_PROMPT,
        description="bge-base-en-v1.5, stronger general-purpose encoder",
    ),
    DenseModelSpec(
        key="specter",
        model_id="sentence-transformers/allenai-specter",
        revision="2c68eeca61259b2dd70c3f2628219f925df7031a",
        description="allenai-specter, scientific-domain encoder",
    ),
)

RERANKER_MODEL_ID = DEFAULT_CROSS_ENCODER_MODEL_ID
RERANKER_MODEL_REVISION = DEFAULT_CROSS_ENCODER_REVISION


def dense_model_to_dict(spec: DenseModelSpec) -> dict[str, str]:
    """Serialize a spec for the run manifest."""
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "query_prompt": spec.query_prompt,
        "description": spec.description,
    }


def reranker_model_to_dict() -> dict[str, str]:
    """Serialize the cross-encoder spec for the run manifest."""
    return {
        "model_id": RERANKER_MODEL_ID,
        "revision": RERANKER_MODEL_REVISION,
    }
