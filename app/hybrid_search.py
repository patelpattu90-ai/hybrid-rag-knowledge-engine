class HybridSearch:

    def __init__(self, semantic_retriever, keyword_retriever):

        self.semantic = semantic_retriever
        self.keyword = keyword_retriever

    def normalize(self, scores):

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def search(self, query, top_k=5):

        semantic_results = self.semantic.search(query)
        keyword_results = self.keyword.search(query)

        semantic_scores = [r["semantic_score"] for r in semantic_results]
        bm25_scores = [r["bm25_score"] for r in keyword_results]

        semantic_scores = self.normalize(semantic_scores)
        bm25_scores = self.normalize(bm25_scores)

        combined = []

        for i in range(len(semantic_results)):

            chunk = semantic_results[i]["chunk"]

            final_score = (
                0.6 * semantic_scores[i] +
                0.4 * bm25_scores[i]
            )

            combined.append((final_score, chunk))

        combined.sort(reverse=True, key=lambda x: x[0])

        return [c[1] for c in combined[:top_k]]