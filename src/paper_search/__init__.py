"""Research paper semantic search package."""

from paper_search.evaluation.metrics import IREvaluator
from paper_search.reranking.reranker import CrossEncoderReranker
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever
from paper_search.retrievers.hybrid import HybridRRFRetriever

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRRFRetriever",
    "IREvaluator",
    "__version__",
]
