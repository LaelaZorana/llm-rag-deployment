"""Tests for VectorRetriever."""

import pytest
import os
import tempfile
from rag.chunker import Chunk
from rag.retriever import VectorRetriever, RetrievalResult


def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, start_idx=0, end_idx=len(text), chunk_id=f"c{idx}", metadata={})


@pytest.fixture
def retriever():
    return VectorRetriever(embedding_dim=64)


@pytest.fixture
def populated_retriever():
    r = VectorRetriever(embedding_dim=64)
    chunks = [
        make_chunk("Machine learning is a subset of AI", 0),
        make_chunk("Python is a popular programming language", 1),
        make_chunk("Deep learning uses neural networks", 2),
        make_chunk("FastAPI is a web framework", 3),
    ]
    r.add_chunks(chunks)
    return r


def test_add_chunks_increases_size(retriever):
    assert retriever.size == 0
    retriever.add_chunks([make_chunk("hello", 0), make_chunk("world", 1)])
    assert retriever.size == 2


def test_retrieve_returns_results(populated_retriever):
    results = populated_retriever.retrieve("machine learning", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_retrieve_top_k_limit(populated_retriever):
    results = populated_retriever.retrieve("anything", top_k=2)
    assert len(results) <= 2


def test_retrieve_empty_index(retriever):
    results = retriever.retrieve("query", top_k=3)
    assert results == []


def test_retrieve_scores_between_neg1_and_1(populated_retriever):
    results = populated_retriever.retrieve("neural network", top_k=3)
    for r in results:
        assert -1.0 <= r.score <= 1.0


def test_save_and_load_index(populated_retriever):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        populated_retriever.save_index(path)
        new_retriever = VectorRetriever(embedding_dim=64)
        new_retriever.load_index(path)
        assert new_retriever.size == populated_retriever.size
        results = new_retriever.retrieve("machine learning", top_k=1)
        assert len(results) == 1
    finally:
        os.unlink(path)


def test_to_dict_serializable(populated_retriever):
    results = populated_retriever.retrieve("python", top_k=1)
    d = results[0].to_dict()
    assert "rank" in d
    assert "score" in d
    assert "text" in d
