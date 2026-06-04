"""
app.py — FastAPI serving layer for the RAG pipeline.

Endpoints:
    POST /ingest  — ingest documents into the index
    POST /query   — query the RAG pipeline
    GET  /health  — liveness check
    GET  /stats   — index size, query count, avg latency

Gracefully skips startup if fastapi/uvicorn not installed.
"""

import sys

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import sys
import os

# Add project root to path when running from serving/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.pipeline import RAGPipeline
from rag.chunker import DocumentChunker
from rag.retriever import VectorRetriever
from rag.generator import RAGGenerator

if not _FASTAPI_AVAILABLE:
    print("FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)


app = FastAPI(
    title="llm-rag-deployment",
    description="Composable RAG pipeline: document ingestion, retrieval, and LLM generation",
    version="0.1.0",
)

# Global pipeline instance (in production, use dependency injection or a singleton factory)
_pipeline = RAGPipeline(
    chunker=DocumentChunker(),
    retriever=VectorRetriever(embedding_dim=384),
    generator=RAGGenerator(model_name="gpt2"),
)


class IngestRequest(BaseModel):
    documents: List[Dict[str, Any]]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int
    index_size: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunks: List[Dict[str, Any]]
    tokens_generated: int
    generation_time_ms: float


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """Ingest documents into the RAG index."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    _pipeline.ingest(request.documents)
    return IngestResponse(
        message=f"Successfully ingested {len(request.documents)} document(s).",
        documents_ingested=len(request.documents),
        index_size=_pipeline.retriever.size,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the RAG pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if _pipeline.retriever.size == 0:
        raise HTTPException(status_code=422, detail="Index is empty. Ingest documents first.")
    response = _pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        source_chunks=[r.to_dict() for r in response.source_chunks],
        tokens_generated=response.tokens_generated,
        generation_time_ms=response.generation_time_ms,
    )


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "index_size": _pipeline.retriever.size}


@app.get("/stats")
def stats():
    """Pipeline statistics."""
    return _pipeline.stats


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
