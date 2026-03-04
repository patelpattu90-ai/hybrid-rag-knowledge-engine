from ingestion import ingest_urls
from chunker import IntelligentChunker
from embeddings import EmbeddingIndexer
from bm25_retriever import BM25Retriever
from hybrid_search import HybridSearch
from reranker import CrossEncoderReranker

if __name__ == "__main__":

    urls = [
        "https://fastapi.tiangolo.com/features/",
        "https://fastapi.tiangolo.com/tutorial/path-operations/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "https://fastapi.tiangolo.com/tutorial/security/",
        "https://fastapi.tiangolo.com/tutorial/middleware/"
    ]

    docs = ingest_urls(
        urls=urls,
        framework="fastapi",
        version="0.110"
    )

    print(f"Sections parsed: {len(docs)}")
   
    chunker = IntelligentChunker(max_tokens=400, overlap=50)
    chunks = chunker.chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}")
    bm25 = BM25Retriever(chunks)
    print(chunks[0])

    indexer = EmbeddingIndexer()

    indexer.build_index(chunks)
    hybrid = HybridSearch(indexer, bm25)

    query = "How does dependency injection work in FastAPI?"

    candidate_chunks, timings = hybrid.search(query, top_k=20)

    reranker = CrossEncoderReranker()

    results = reranker.rerank(query, candidate_chunks, top_k=5)

    print("\nRetrieval Latency (ms):")
    print(timings)
    
    print("\nTop Results After Re-Ranking:\n")
    for r in results:
        print(r["section_title"])