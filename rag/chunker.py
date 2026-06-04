"""
chunker.py: Document chunking strategies for RAG pipelines.

Supports three strategies:
- fixed: fixed-size token windows with overlap
- sentence: split on sentence boundaries
- paragraph: split on paragraph/double-newline boundaries
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "metadata": self.metadata,
        }


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def _chunk_fixed(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: Optional[Dict] = None,
) -> List[Chunk]:
    tokens = _tokenize(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        # Approximate char offsets
        start_idx = len(" ".join(tokens[:i])) + (1 if i > 0 else 0)
        end_idx = start_idx + len(chunk_text)
        chunks.append(
            Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_id=str(uuid.uuid4()),
                metadata=dict(metadata or {}),
            )
        )
        i += step
    return chunks


def _chunk_sentence(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: Optional[Dict] = None,
) -> List[Chunk]:
    # Split on sentence-ending punctuation
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)

    chunks = []
    current_tokens: List[str] = []
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        s_tokens = _tokenize(sentence)
        if len(current_tokens) + len(s_tokens) > chunk_size and current_tokens:
            chunk_text = " ".join(current_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=current_start,
                    end_idx=current_start + len(chunk_text),
                    chunk_id=str(uuid.uuid4()),
                    metadata=dict(metadata or {}),
                )
            )
            # Overlap: keep last `overlap` tokens
            current_tokens = current_tokens[-overlap:] if overlap > 0 else []
            current_start = char_pos - len(" ".join(current_tokens))

        current_tokens.extend(s_tokens)
        char_pos += len(sentence) + 1

    if current_tokens:
        chunk_text = " ".join(current_tokens)
        chunks.append(
            Chunk(
                text=chunk_text,
                start_idx=current_start,
                end_idx=current_start + len(chunk_text),
                chunk_id=str(uuid.uuid4()),
                metadata=dict(metadata or {}),
            )
        )

    return chunks if chunks else [Chunk(text=text, start_idx=0, end_idx=len(text), chunk_id=str(uuid.uuid4()), metadata=dict(metadata or {}))]


def _chunk_paragraph(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: Optional[Dict] = None,
) -> List[Chunk]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_tokens: List[str] = []
    current_start = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2
            continue
        p_tokens = _tokenize(para)
        if len(current_tokens) + len(p_tokens) > chunk_size and current_tokens:
            chunk_text = " ".join(current_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_idx=current_start,
                    end_idx=current_start + len(chunk_text),
                    chunk_id=str(uuid.uuid4()),
                    metadata=dict(metadata or {}),
                )
            )
            current_tokens = current_tokens[-overlap:] if overlap > 0 else []
            current_start = char_pos

        current_tokens.extend(p_tokens)
        char_pos += len(para) + 2

    if current_tokens:
        chunk_text = " ".join(current_tokens)
        chunks.append(
            Chunk(
                text=chunk_text,
                start_idx=current_start,
                end_idx=current_start + len(chunk_text),
                chunk_id=str(uuid.uuid4()),
                metadata=dict(metadata or {}),
            )
        )

    return chunks if chunks else [Chunk(text=text, start_idx=0, end_idx=len(text), chunk_id=str(uuid.uuid4()), metadata=dict(metadata or {}))]


_STRATEGIES = {
    "fixed": _chunk_fixed,
    "sentence": _chunk_sentence,
    "paragraph": _chunk_paragraph,
}


class DocumentChunker:
    """
    Chunks text documents into overlapping windows for RAG retrieval.

    Usage:
        chunker = DocumentChunker()
        chunks = chunker.chunk(text, strategy='sentence', chunk_size=256, overlap=32)
    """

    def chunk(
        self,
        text: str,
        strategy: str = "fixed",
        chunk_size: int = 512,
        overlap: int = 64,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a single text string.

        Args:
            text: Input text to chunk.
            strategy: 'fixed', 'sentence', or 'paragraph'.
            chunk_size: Target chunk size in tokens.
            overlap: Number of tokens to overlap between chunks.
            metadata: Optional dict attached to each chunk.

        Returns:
            List of Chunk objects.
        """
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(_STRATEGIES.keys())}")
        if not text or not text.strip():
            return []
        return _STRATEGIES[strategy](text, chunk_size=chunk_size, overlap=overlap, metadata=metadata)

    def chunk_documents(self, docs: List[Dict]) -> List[Chunk]:
        """
        Chunk a list of document dicts.

        Args:
            docs: List of dicts with at least 'text' key. Optional keys:
                  'strategy', 'chunk_size', 'overlap', 'metadata'.

        Returns:
            Flat list of Chunk objects from all documents.
        """
        all_chunks = []
        for i, doc in enumerate(docs):
            text = doc.get("text", "")
            strategy = doc.get("strategy", "fixed")
            chunk_size = doc.get("chunk_size", 512)
            overlap = doc.get("overlap", 64)
            metadata = doc.get("metadata", {})
            metadata.setdefault("doc_index", i)
            metadata.setdefault("doc_id", doc.get("id", str(i)))
            chunks = self.chunk(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap, metadata=metadata)
            all_chunks.extend(chunks)
        return all_chunks
