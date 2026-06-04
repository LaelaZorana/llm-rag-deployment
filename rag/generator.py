"""
generator.py: LLM-based answer generation for RAG pipelines.

Uses HuggingFace transformers pipeline with graceful fallback if not installed.
The prompt template puts retrieved context before the question so the model
can attend to evidence when generating the answer.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

from .retriever import RetrievalResult

import os as _os

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def _is_test_environment() -> bool:
    """Return True if running under pytest, to avoid loading heavy models in tests."""
    return "PYTEST_CURRENT_TEST" in _os.environ or "pytest" in _os.environ.get("_", "")


@dataclass
class GenerationResult:
    answer: str
    source_chunks: List[RetrievalResult]
    prompt_used: str
    tokens_generated: int
    generation_time_ms: float

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "source_chunks": [r.to_dict() for r in self.source_chunks],
            "prompt_used": self.prompt_used[:200] + "..." if len(self.prompt_used) > 200 else self.prompt_used,
            "tokens_generated": self.tokens_generated,
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


_RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant. Use the context below to answer the question.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""


class RAGGenerator:
    """
    Wraps a HuggingFace text-generation pipeline for RAG answer generation.

    Falls back to a template-based stub if transformers is not installed,
    so the rest of the pipeline stays functional in test environments.

    Usage:
        generator = RAGGenerator(model_name='gpt2')
        result = generator.generate(query, context_chunks, max_new_tokens=128)
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._pipeline: Optional[Any] = None

        if _TRANSFORMERS_AVAILABLE and not _is_test_environment():
            try:
                self._pipeline = hf_pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1 if device == "cpu" else 0,
                )
            except Exception as e:
                # Model download failed or other init error, degrade gracefully
                self._pipeline = None
                self._init_error = str(e)

    def format_prompt(self, query: str, chunks: List[RetrievalResult]) -> str:
        """
        Build the RAG prompt from query and retrieved context chunks.

        Args:
            query: The user's question.
            chunks: Retrieved context chunks sorted by relevance.

        Returns:
            Formatted prompt string.
        """
        context_parts = []
        for i, result in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {result.chunk.text.strip()}")
        context = "\n\n".join(context_parts)
        return _RAG_PROMPT_TEMPLATE.format(context=context, query=query)

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query: The question to answer.
            context_chunks: Retrieved chunks from VectorRetriever.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with answer, sources, and timing.
        """
        prompt = self.format_prompt(query, context_chunks)
        start = time.time()

        if self._pipeline is not None:
            try:
                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._pipeline.tokenizer.eos_token_id,
                )
                generated_text = outputs[0]["generated_text"]
                # Strip the prompt prefix to get just the new tokens
                answer = generated_text[len(prompt):].strip()
                tokens_generated = len(answer.split())
            except Exception as e:
                answer = f"[Generation error: {e}]"
                tokens_generated = 0
        else:
            # Fallback stub when transformers not available
            answer = (
                f"[Stub answer: install transformers for real generation] "
                f"Based on {len(context_chunks)} retrieved chunk(s), this is a placeholder response."
            )
            tokens_generated = len(answer.split())

        elapsed_ms = (time.time() - start) * 1000

        return GenerationResult(
            answer=answer,
            source_chunks=context_chunks,
            prompt_used=prompt,
            tokens_generated=tokens_generated,
            generation_time_ms=elapsed_ms,
        )
