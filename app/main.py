import os
from dotenv import load_dotenv
from generator import GroqGenerator
from ingestion import ingest_urls
from chunker import IntelligentChunker
from embeddings import EmbeddingIndexer
from bm25_retriever import BM25Retriever
from hybrid_search import HybridSearch
from reranker import CrossEncoderReranker
from evaluator import RetrievalEvaluator
from eval_dataset import eval_queries

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if __name__ == "__main__":

    urls = [
        "https://fastapi.tiangolo.com/features/",
        "https://fastapi.tiangolo.com/tutorial/path-operations/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "https://fastapi.tiangolo.com/tutorial/security/",
        "https://fastapi.tiangolo.com/tutorial/middleware/"
    ]
    # Ingestion call
    docs = ingest_urls(
        urls=urls,
        framework="fastapi",
        version="0.110"
    )

    print(f"Sections parsed: {len(docs)}")

#    Chunking call
    chunker = IntelligentChunker(max_tokens=400, overlap=50)
    chunks = chunker.chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    # Keyword index
    bm25 = BM25Retriever(chunks)
    print(chunks[0])
 
    # Call vector index
    indexer = EmbeddingIndexer()

    indexer.build_index(chunks)
    
    # Call Hybrid search 
    hybrid = HybridSearch(indexer, bm25)

    query = "How does dependency injection work in FastAPI?"

    candidate_chunks, timings = hybrid.search(query, top_k=20)

    reranker = CrossEncoderReranker()

    results = reranker.rerank(query, candidate_chunks, top_k=5)

    print("\nRetrieval Latency (ms):")
    print(timings)
    
    print("\nTop Results After Re-Ranking:\n")
    
    generator = GroqGenerator(groq_key)

    answer = generator.generate(query, results)

    print("\n===== GENERATED ANSWER =====\n")
    print(answer)
    print(results[0])

from eval_dataset import eval_queries
from evaluator import RetrievalEvaluator

evaluator = RetrievalEvaluator()

precision_scores = []
mrr_scores = []
ndcg_scores = []

for item in eval_queries:

    query = item["query"]
    relevant = item["relevant_sections"]

    results, latency = hybrid.search(query)

    p = evaluator.precision_at_k(results, relevant, k=5)
    m = evaluator.mrr(results, relevant)
    n = evaluator.ndcg(results, relevant, k=5)

    precision_scores.append(p)
    mrr_scores.append(m)
    ndcg_scores.append(n)

avg_precision = sum(precision_scores) / len(precision_scores)
avg_mrr = sum(mrr_scores) / len(mrr_scores)
avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

print("\n===== RETRIEVAL EVALUATION =====")
print(f"Queries evaluated: {len(eval_queries)}")
print(f"Average Precision@5: {avg_precision:.2f}")
print(f"Average MRR: {avg_mrr:.2f}")
print(f"Average nDCG@5: {avg_ndcg:.2f}")

import matplotlib.pyplot as plt

metrics = {
    "Precision@5": avg_precision,
    "MRR": avg_mrr,
    "nDCG@5": avg_ndcg
}

plt.bar(metrics.keys(), metrics.values())
plt.title("RAG Retrieval Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.show()