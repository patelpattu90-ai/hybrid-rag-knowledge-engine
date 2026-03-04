from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        corpus = [c["text"].split() for c in chunks]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        results = []

        for idx, score in enumerate(scores):

            results.append({
                "chunk": self.chunks[idx],
                "bm25_score": score
            })

        return results