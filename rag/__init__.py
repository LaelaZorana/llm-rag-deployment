"""
llm-rag-deployment: Composable RAG pipeline: chunking, retrieval, generation, serving.
"""

from .chunker import DocumentChunker, Chunk
from .retriever import VectorRetriever, RetrievalResult
from .generator import RAGGenerator, GenerationResult
from .pipeline import RAGPipeline, RAGResponse, EvaluationReport

__version__ = "0.1.0"
__all__ = [
    "DocumentChunker",
    "Chunk",
    "VectorRetriever",
    "RetrievalResult",
    "RAGGenerator",
    "GenerationResult",
    "RAGPipeline",
    "RAGResponse",
    "EvaluationReport",
]
