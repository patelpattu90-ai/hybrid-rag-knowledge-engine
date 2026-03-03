from ingestion import ingest_urls
from chunker import IntelligentChunker

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
    print(chunks[0])