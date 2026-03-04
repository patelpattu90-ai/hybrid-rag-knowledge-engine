from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self):

        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks, top_k=5):

        pairs = []

        for c in chunks:
            pairs.append((query, c["text"]))

        scores = self.model.predict(pairs)

        ranked = list(zip(scores, chunks))

        ranked.sort(reverse=True, key=lambda x: x[0])

        results = [r[1] for r in ranked[:top_k]]

        return results