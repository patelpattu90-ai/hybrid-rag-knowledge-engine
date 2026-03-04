from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class EmbeddingIndexer:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.text_chunks = chunks

    def search(self, query, top_k=5):

        query_embedding = self.model.encode([query])

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results