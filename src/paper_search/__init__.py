"""Research paper semantic search package."""

from paper_search.evaluation.metrics import IREvaluator
from paper_search.reranking.reranker import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    PairScorer,
)
from paper_search.retrievers.base import Retriever
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import (
    HashedTfidfRetriever,
    SentenceTransformerEncoder,
    SentenceTransformerRetriever,
    TextEncoder,
)
from paper_search.retrievers.hybrid import HybridRRFRetriever

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "HashedTfidfRetriever",
    "HybridRRFRetriever",
    "IREvaluator",
    "LexicalOverlapReranker",
    "PairScorer",
    "Retriever",
    "SentenceTransformerEncoder",
    "SentenceTransformerRetriever",
    "TextEncoder",
    "__version__",
]
