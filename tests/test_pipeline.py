"""Tests for RAGPipeline."""

import pytest
from rag.pipeline import RAGPipeline, RAGResponse, EvaluationReport
from rag.chunker import DocumentChunker
from rag.retriever import VectorRetriever
from rag.generator import RAGGenerator


DOCS = [
    {"id": "d1", "text": "Machine learning enables computers to learn from data without explicit programming."},
    {"id": "d2", "text": "Python is widely used for data science, machine learning, and web development."},
    {"id": "d3", "text": "Neural networks are inspired by the structure of biological brains."},
    {"id": "d4", "text": "FastAPI is a modern web framework for building APIs with Python."},
]


@pytest.fixture
def pipeline():
    p = RAGPipeline(
        chunker=DocumentChunker(),
        retriever=VectorRetriever(embedding_dim=64),
        generator=RAGGenerator(),
    )
    p.ingest(DOCS)
    return p


def test_ingest_builds_index(pipeline):
    assert pipeline.retriever.size > 0


def test_query_returns_rag_response(pipeline):
    response = pipeline.query("What is machine learning?")
    assert isinstance(response, RAGResponse)


def test_query_has_answer(pipeline):
    response = pipeline.query("Tell me about Python")
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0


def test_query_has_source_chunks(pipeline):
    response = pipeline.query("neural networks", top_k=2)
    assert len(response.source_chunks) <= 2
    assert len(response.source_chunks) > 0


def test_empty_index_before_ingest():
    p = RAGPipeline()
    response = p.query("anything")
    # Should return a response even with empty index (no results retrieved)
    assert isinstance(response, RAGResponse)


def test_evaluate_returns_report(pipeline):
    qa_pairs = [
        {"question": "What is Python?", "expected_answer": "Python is a programming language."},
        {"question": "What is ML?", "expected_answer": "Machine learning"},
    ]
    report = pipeline.evaluate(qa_pairs)
    assert isinstance(report, EvaluationReport)
    assert report.total_questions == 2


def test_stats_tracked(pipeline):
    pipeline.query("test query")
    stats = pipeline.stats
    assert stats["total_queries"] >= 1
    assert stats["index_size"] > 0


def test_to_dict_serializable(pipeline):
    response = pipeline.query("test")
    d = response.to_dict()
    assert "question" in d
    assert "answer" in d
    assert "source_chunks" in d
