# llm-rag-deployment

Every LLM project I worked on eventually needed a retrieval layer — and I kept reinventing the same chunking/retrieval/generation pipeline. This is my clean, composable version that I can drop into any project.

The design goal was zero magic: each component (chunker, retriever, generator) is a standalone class you can test and swap independently. The FastAPI layer is thin. The only required dependency is numpy.

## Architecture

```
Documents
    │
    ▼
┌─────────────────┐
│ DocumentChunker │  fixed / sentence / paragraph strategies
│                 │  configurable size + overlap
└────────┬────────┘
         │ chunks
         ▼
┌─────────────────┐
│ VectorRetriever │  in-memory cosine similarity index
│                 │  save/load with pickle
│  (numpy only)   │  swap _embed() for real model
└────────┬────────┘
         │ top-k results
         ▼
┌─────────────────┐
│  RAGGenerator   │  HuggingFace pipeline wrapper
│                 │  graceful stub if transformers not installed
└────────┬────────┘
         │ answer + sources
         ▼
┌─────────────────┐
│   RAGPipeline   │  compose all three + evaluate()
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI app   │  POST /ingest  POST /query  GET /stats
└─────────────────┘
```

## Quickstart

```bash
git clone https://github.com/LaelaZorana/llm-rag-deployment
cd llm-rag-deployment
pip install -r requirements.txt
```

**Run the demo:**
```bash
python examples/demo_pipeline.py
```

**Use the pipeline in code:**
```python
from rag import RAGPipeline, DocumentChunker, VectorRetriever, RAGGenerator

pipeline = RAGPipeline(
    chunker=DocumentChunker(),
    retriever=VectorRetriever(embedding_dim=384),
    generator=RAGGenerator(),
)

pipeline.ingest([
    {"id": "doc1", "text": "Your document text here..."},
    {"id": "doc2", "text": "Another document..."},
])

response = pipeline.query("What does doc1 talk about?")
print(response.answer)
print("Sources:", [r.chunk.metadata["doc_id"] for r in response.source_chunks])
```

**Chunking strategies:**
```python
from rag import DocumentChunker

chunker = DocumentChunker()
chunks = chunker.chunk(text, strategy="sentence", chunk_size=256, overlap=32)
# strategies: 'fixed', 'sentence', 'paragraph'
```

**Evaluate on QA pairs:**
```python
report = pipeline.evaluate([
    {"question": "What is X?", "expected_answer": "X is..."},
])
print(report.summary())  # exact match + keyword overlap metrics
```

**Run tests:**
```bash
python -m pytest tests/ -q
```

## FastAPI Serving

```bash
pip install fastapi uvicorn transformers torch
uvicorn serving.app:app --reload
```

```bash
# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"id": "1", "text": "Your content here"}]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "top_k": 3}'
```

## Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 rag-api
```

## Note on Embeddings

The retriever uses deterministic random vectors as placeholder embeddings (seeded from text hash). This lets everything work with numpy only. For production, replace `VectorRetriever._embed()` with a real model:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

class MyRetriever(VectorRetriever):
    def _embed(self, text):
        return model.encode(text)
```

## About

Applied ML and cloud engineer building production AI systems.

- GitHub: [github.com/LaelaZorana](https://github.com/LaelaZorana)
- HuggingFace: [huggingface.co/LaelaZ](https://huggingface.co/LaelaZ)
- Kaggle: [kaggle.com/laelazorana](https://www.kaggle.com/laelazorana)

MIT License.
