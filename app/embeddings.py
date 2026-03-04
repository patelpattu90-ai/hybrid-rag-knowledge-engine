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


    def search(self, query):

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, len(self.text_chunks))

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "chunk": self.text_chunks[idx],
                "semantic_score": float(score)
            })

        return results