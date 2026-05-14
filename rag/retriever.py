"""
retriever.py — In-memory vector retrieval using cosine similarity.

NOTE: This uses random numpy embeddings as a placeholder. In production you'd
swap in real embeddings from sentence-transformers, OpenAI, or similar. The
architecture stays the same — just replace _embed() with a real model call.

Why no external vector DB? For portability. This runs with numpy only and
makes the retrieval logic easy to unit test without spinning up infrastructure.
"""

import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .chunker import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorRetriever:
    """
    Simple in-memory vector index with cosine similarity retrieval.

    NOTE: Embeddings are random numpy vectors seeded from text hash.
    Replace _embed() with a real embedding model for production use.

    Usage:
        retriever = VectorRetriever(embedding_dim=384)
        retriever.add_chunks(chunks)
        results = retriever.retrieve("what is machine learning?", top_k=3)
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def _embed(self, text: str) -> np.ndarray:
        """
        Placeholder embedding: deterministic random vector seeded from text hash.

        In production, replace with:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text)
        """
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Index a list of chunks.

        Args:
            chunks: List of Chunk objects to add to the index.
        """
        if not chunks:
            return
        new_embeddings = np.stack([self._embed(c.text) for c in chunks])
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        self._chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks by cosine similarity to query.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if not self._chunks or self._embeddings is None:
            return []

        query_vec = self._embed(query)
        # Batch cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
        normed = self._embeddings / norms
        scores = normed @ query_vec

        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results

    def save_index(self, path: str) -> None:
        """Serialize the index to disk with pickle."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "embeddings": self._embeddings,
                    "embedding_dim": self.embedding_dim,
                },
                f,
            )

    def load_index(self, path: str) -> None:
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self.embedding_dim = data["embedding_dim"]

    @property
    def size(self) -> int:
        return len(self._chunks)
