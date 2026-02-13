# 📊 RAG Tutorial 09 — Re-ranking Pipeline

<p align="center">
  <a href="https://github.com/kishore2797/mastering-rag"><img src="https://img.shields.io/badge/Series-Mastering_RAG-blue?style=for-the-badge" /></a>
  <img src="https://img.shields.io/badge/Part-9_of_16-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Difficulty-Advanced-orange?style=for-the-badge" />
</p>

> **Part of the [Mastering RAG](https://github.com/kishore2797/mastering-rag) tutorial series**  
> Previous: [08 — Query Rewriting](https://github.com/kishore2797/rag-08-query-rewriting) | Next: [10 — Self-Reflective RAG](https://github.com/kishore2797/rag-10-self-reflective-rag)

---

## 🌍 Real-World Scenario

> An e-commerce search returns 50 products for "wireless noise-cancelling headphones for running." Vector search finds headphones, but position #1 is a wired studio headphone (semantically similar to "noise-cancelling"). After **cross-encoder re-ranking** — which reads the full query AND each product description together — the truly relevant wireless sport headphones jump to the top. Re-ranking turns "close enough" into "exactly right."

---

## 🏗️ What You'll Build

A two-stage retrieval pipeline: fast **vector search** returns N candidates, then a **cross-encoder re-ranker** (Cohere Rerank API or local BGE-reranker) reorders them by true relevance, keeping only the top K for the LLM.

```
Query ──→ Vector Search (top 50) ──→ Cross-Encoder Rerank ──→ Top 5 ──→ LLM
           Fast, approximate           Slow, precise            Best chunks
```

## 🔑 Key Concepts

- **Bi-encoder vs. cross-encoder**: bi-encoders are fast (encode independently), cross-encoders are accurate (encode query+doc together)
- **Two-stage retrieval**: use the fast model for recall, precise model for ranking
- **Cohere Rerank**: cloud API, easy to use, strong performance
- **BGE-reranker**: local, free, no API dependency
- **N and K tuning**: retrieve N, rerank to K — both impact quality and latency

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+ · FastAPI · ChromaDB · Cohere · BGE-reranker · Sentence-Transformers |
| Frontend | React 19 · Vite · Tailwind CSS |

## 🚀 Quick Start

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # COHERE_API_KEY for Cohere; BGE runs locally
uvicorn app.main:app --reload --port 8002
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5176 — ingest documents, query with rerank toggle, see before/after order.

## 📖 What You'll Learn

1. Why vector search alone isn't enough for top-quality retrieval
2. How cross-encoders work differently from bi-encoders
3. When to use Cohere (cloud, easy) vs. BGE (local, free)
4. How to tune N (candidates) and K (final results)
5. Measuring the quality improvement from re-ranking

## 📋 Prerequisites

- Python 3.11+ and Node.js 18+
- Concepts from [Tutorial 05](https://github.com/kishore2797/rag-05-basic-rag-pipeline) (basic RAG pipeline)
- Cohere API key (for Cohere Rerank) OR local GPU/CPU for BGE-reranker

## ✏️ Exercises

1. **N and K sweep**: Fix K=5. Try N=10, 20, 50, 100. Plot retrieval quality vs. latency. Find the sweet spot.
2. **Provider comparison**: Run the same 20 queries through Cohere Rerank and local BGE-reranker. Compare quality scores and latency.
3. **Before/after analysis**: For each query, log the rank positions before and after reranking. How many queries see their best chunk move from position 10+ to top-3?
4. **Combined pipeline**: Add re-ranking to a pipeline that already uses query rewriting (Tutorial 08). Do the improvements stack?
5. **Failure cases**: Find queries where re-ranking makes results *worse*. Why does it happen? (Hint: when the original query is ambiguous)

## ⚠️ Common Mistakes

| Mistake | Why It Happens | How to Fix |
|---------|---------------|------------|
| Re-ranking is too slow | Sending 100+ candidates to a cross-encoder | Reduce N to 20–30; the cross-encoder reads every pair (query, doc) |
| BGE-reranker OOMs on large batches | Cross-encoders are memory-hungry | Process in smaller batches (8–16 at a time) or use a smaller model |
| Reranking scores don't match expectations | Different rerankers use different score scales | Normalize scores to [0, 1] before comparing across providers |
| No quality improvement from reranking | Your top-5 from vector search was already good | Reranking helps most when recall is high but precision is low (N >> K) |

## 📚 Further Reading

- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking) — API docs and best practices
- [Cross-Encoders for Re-Ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html) — Sentence-Transformers guide
- [BGE-Reranker Models](https://huggingface.co/BAAI/bge-reranker-large) — Open-source reranker models
- [The Role of Rerankers in RAG](https://www.pinecone.io/learn/series/rag/rerankers/) — Pinecone's explanation of two-stage retrieval

## ➡️ Next Steps

Head to **[Tutorial 10 — Self-Reflective RAG](https://github.com/kishore2797/rag-10-self-reflective-rag)** to add an intelligent agent that evaluates retrieval quality before answering.

---

<p align="center">
  <sub>Part of <a href="https://github.com/kishore2797/mastering-rag">Mastering RAG — From Zero to Production</a></sub>
</p>
