"""
pipeline.py — End-to-end RAG pipeline composing chunker, retriever, and generator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .chunker import DocumentChunker, Chunk
from .retriever import VectorRetriever, RetrievalResult
from .generator import RAGGenerator, GenerationResult


@dataclass
class RAGResponse:
    question: str
    answer: str
    source_chunks: List[RetrievalResult]
    tokens_generated: int
    generation_time_ms: float

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_chunks": [r.to_dict() for r in self.source_chunks],
            "tokens_generated": self.tokens_generated,
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


@dataclass
class EvaluationReport:
    total_questions: int
    exact_matches: int
    keyword_overlaps: List[float]
    avg_keyword_overlap: float
    results: List[Dict[str, Any]]

    def summary(self) -> str:
        return (
            f"Evaluation: {self.total_questions} questions, "
            f"{self.exact_matches} exact matches, "
            f"avg keyword overlap: {self.avg_keyword_overlap:.2%}"
        )


def _keyword_overlap(predicted: str, expected: str) -> float:
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    return len(pred_words & exp_words) / len(exp_words)


class RAGPipeline:
    """
    Composes DocumentChunker, VectorRetriever, and RAGGenerator into a single pipeline.

    Usage:
        pipeline = RAGPipeline(chunker, retriever, generator)
        pipeline.ingest(documents)
        response = pipeline.query("What is machine learning?")
    """

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        retriever: Optional[VectorRetriever] = None,
        generator: Optional[RAGGenerator] = None,
    ):
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or VectorRetriever()
        self.generator = generator or RAGGenerator()
        self._total_queries = 0
        self._total_latency_ms = 0.0

    def ingest(self, documents: List[Dict]) -> None:
        """
        Chunk and index all documents.

        Args:
            documents: List of dicts with 'text' key and optional metadata.
        """
        chunks = self.chunker.chunk_documents(documents)
        self.retriever.add_chunks(chunks)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """
        Run end-to-end RAG: retrieve context, generate answer.

        Args:
            question: The user's question.
            top_k: Number of context chunks to retrieve.

        Returns:
            RAGResponse with answer and sources.
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        gen_result = self.generator.generate(question, retrieved)

        self._total_queries += 1
        self._total_latency_ms += gen_result.generation_time_ms

        return RAGResponse(
            question=question,
            answer=gen_result.answer,
            source_chunks=retrieved,
            tokens_generated=gen_result.tokens_generated,
            generation_time_ms=gen_result.generation_time_ms,
        )

    def evaluate(self, qa_pairs: List[Dict]) -> EvaluationReport:
        """
        Evaluate pipeline on a list of question/answer pairs.

        Args:
            qa_pairs: List of dicts with 'question' and 'expected_answer' keys.

        Returns:
            EvaluationReport with exact match and keyword overlap metrics.
        """
        results = []
        exact_matches = 0
        overlaps = []

        for pair in qa_pairs:
            question = pair["question"]
            expected = pair.get("expected_answer", "")
            response = self.query(question)
            exact = response.answer.strip().lower() == expected.strip().lower()
            overlap = _keyword_overlap(response.answer, expected)
            if exact:
                exact_matches += 1
            overlaps.append(overlap)
            results.append({
                "question": question,
                "expected": expected,
                "predicted": response.answer,
                "exact_match": exact,
                "keyword_overlap": overlap,
            })

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        return EvaluationReport(
            total_questions=len(qa_pairs),
            exact_matches=exact_matches,
            keyword_overlaps=overlaps,
            avg_keyword_overlap=avg_overlap,
            results=results,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "index_size": self.retriever.size,
            "total_queries": self._total_queries,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
        }
