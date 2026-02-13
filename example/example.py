#!/usr/bin/env python3
"""
RAG Tutorial 09 — Re-ranking Pipeline
Minimal example: retrieve top-N with vector search, then re-rank with a cross-encoder to top-K.
Run: pip install -r requirements.txt && python example.py
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def main():
    docs = [
        "RAG uses retrieval then generation.",
        "Vector search finds similar chunks quickly.",
        "Re-ranking improves precision by scoring query-document pairs with a cross-encoder.",
        "The best pasta recipe uses fresh tomatoes and basil.",
    ]
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("rerank_example")
    coll.add(
        ids=[f"d_{i}" for i in range(len(docs))],
        embeddings=bi_encoder.encode(docs).tolist(),
        documents=docs,
    )
    query = "How does re-ranking work in RAG?"
    # Stage 1: vector search, get more candidates (e.g. N=4)
    results = coll.query(
        query_embeddings=bi_encoder.encode([query]).tolist(),
        n_results=4,
        include=["documents"],
    )
    candidates = results["documents"][0]
    # Stage 2: cross-encoder re-rank to top K=2
    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    top_k = [doc for _, doc in ranked[:2]]
    print("Query:", query)
    print("After vector search (4):", candidates)
    print("After re-rank (top 2):", top_k)
    print("\n→ Cross-encoder puts the most relevant chunk on top.")


if __name__ == "__main__":
    main()
