---
title: Hybrid RAG Knowledge Engine
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: true
license: mit
short_description: Production RAG system on FastAPI docs — 10 pipeline layers
---

# ⚡ Hybrid RAG Knowledge Engine

A production-grade Retrieval-Augmented Generation system built on FastAPI documentation.

**10 pipeline layers:** Semantic + BM25 hybrid search · Cross-encoder reranking · LLM generation · Streaming · Cache · Query Rewriting · Memory · Tool Routing · Guardrails · Security · RAGAS Evaluation · Doc Quality · Contextual Compression

Built with LLaMA 3.3 70B via Groq · FAISS · BM25 · SentenceTransformers · Gradio

---

## How to use

Ask any question about FastAPI:
- `How do path parameters work?`
- `Give me an overview of error handling in FastAPI`
- `What is dependency injection?`

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Keyword Search | BM25Okapi |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| UI | Gradio |

## Source

GitHub: [patelpattu90-ai/hybrid-rag-knowledge-engine](https://github.com/patelpattu90-ai/hybrid-rag-knowledge-engine)
