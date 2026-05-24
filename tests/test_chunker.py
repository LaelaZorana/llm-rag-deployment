"""Tests for DocumentChunker."""

import pytest
from rag.chunker import DocumentChunker, Chunk

LONG_TEXT = " ".join([f"word{i}" for i in range(600)])
SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
PARA_TEXT = "First paragraph with some content here.\n\nSecond paragraph with different content.\n\nThird paragraph."


@pytest.fixture
def chunker():
    return DocumentChunker()


def test_fixed_chunking_returns_list(chunker):
    chunks = chunker.chunk(LONG_TEXT, strategy="fixed", chunk_size=100, overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_fixed_chunk_size(chunker):
    chunks = chunker.chunk(LONG_TEXT, strategy="fixed", chunk_size=100, overlap=0)
    for c in chunks[:-1]:
        assert len(c.text.split()) <= 100


def test_sentence_chunking(chunker):
    text = "This is sentence one. This is sentence two. This is sentence three. " * 20
    chunks = chunker.chunk(text, strategy="sentence", chunk_size=50)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_paragraph_chunking(chunker):
    chunks = chunker.chunk(PARA_TEXT, strategy="paragraph", chunk_size=20)
    assert len(chunks) > 0


def test_empty_text_returns_empty(chunker):
    assert chunker.chunk("", strategy="fixed") == []
    assert chunker.chunk("   ", strategy="fixed") == []


def test_unknown_strategy_raises(chunker):
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunker.chunk("some text", strategy="magic")


def test_chunk_has_required_fields(chunker):
    chunks = chunker.chunk(SHORT_TEXT, strategy="fixed", chunk_size=10)
    for c in chunks:
        assert hasattr(c, "text")
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "start_idx")
        assert hasattr(c, "end_idx")
        assert hasattr(c, "metadata")


def test_chunk_documents_batch(chunker):
    docs = [
        {"id": "a", "text": LONG_TEXT},
        {"id": "b", "text": SHORT_TEXT},
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 2
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert "a" in doc_ids and "b" in doc_ids
