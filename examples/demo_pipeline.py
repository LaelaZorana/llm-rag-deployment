"""
demo_pipeline.py — End-to-end demonstration of the RAG pipeline with synthetic documents.

Run with:
    python examples/demo_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import RAGPipeline, DocumentChunker, VectorRetriever, RAGGenerator

DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. "
            "It focuses on developing computer programs that can access data and use it to learn for themselves. "
            "The process begins with observations or data, such as examples, direct experience, or instruction."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
            "A retriever fetches relevant documents from a knowledge base, and a language model generates "
            "an answer conditioned on the retrieved context. This approach reduces hallucination and "
            "enables the model to cite sources. RAG is widely used for question answering and chatbots."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Vector databases store high-dimensional embeddings and support fast approximate nearest neighbor search. "
            "Popular options include Pinecone, Weaviate, Chroma, and pgvector. "
            "They're the backbone of modern RAG systems — instead of keyword search, "
            "semantic similarity lets you retrieve conceptually related content."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Fine-tuning a language model adapts a pretrained model to a specific task or domain. "
            "Techniques like LoRA (Low-Rank Adaptation) and QLoRA make fine-tuning feasible on consumer hardware "
            "by reducing the number of trainable parameters. RLHF (Reinforcement Learning from Human Feedback) "
            "is used to align models with human preferences."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "MLOps combines machine learning, DevOps, and data engineering to streamline ML deployment. "
            "Key practices include model versioning, automated testing, CI/CD pipelines for models, "
            "monitoring for data drift and model degradation, and reproducible training pipelines. "
            "Tools like MLflow, BentoML, and Ray Serve are commonly used."
        ),
    },
]

QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does fine-tuning work with LoRA?",
    "What are vector databases used for?",
]


def main():
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    chunker = DocumentChunker()
    retriever = VectorRetriever(embedding_dim=384)
    generator = RAGGenerator()  # stub mode if transformers not installed

    pipeline = RAGPipeline(chunker=chunker, retriever=retriever, generator=generator)

    print(f"\nIngesting {len(DOCUMENTS)} documents...")
    pipeline.ingest(DOCUMENTS)
    print(f"Index size: {pipeline.retriever.size} chunks")

    for question in QUESTIONS:
        print(f"\n{'─' * 50}")
        print(f"Q: {question}")
        response = pipeline.query(question, top_k=2)
        print(f"A: {response.answer[:200]}...")
        print(f"   Sources: {[r.chunk.metadata.get('doc_id') for r in response.source_chunks]}")
        print(f"   Latency: {response.generation_time_ms:.0f}ms")

    print(f"\n{'─' * 50}")
    print("Pipeline stats:", pipeline.stats)


if __name__ == "__main__":
    main()
