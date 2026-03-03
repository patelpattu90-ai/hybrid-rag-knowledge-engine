import tiktoken
from typing import List, Dict
import uuid


class IntelligentChunker:
    def __init__(self, max_tokens: int = 400, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.max_tokens - self.overlap

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunked_docs = []

        for doc in documents:
            text_chunks = self.split_text(doc["text"])

            for idx, chunk in enumerate(text_chunks):
                chunked_docs.append({
                    "chunk_id": str(uuid.uuid4()),
                    "framework": doc["framework"],
                    "version": doc["version"],
                    "source_url": doc["source_url"],
                    "section_title": doc["section_title"],
                    "chunk_index": idx,
                    "text": chunk
                })

        return chunked_docs