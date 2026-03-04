import math

class RetrievalEvaluator:

    def normalize(self, text):
        return text.lower().replace("]", "").replace('"', "").strip()


    def is_relevant(self, title, relevant_sections):

        title = self.normalize(title)

        for section in relevant_sections:
            if self.normalize(section) in title:
                return True

        return False


    def precision_at_k(self, retrieved_chunks, relevant_sections, k=5):

        retrieved = retrieved_chunks[:k]
        relevant_found = 0

        for r in retrieved:
            if self.is_relevant(r["section_title"], relevant_sections):
                relevant_found += 1

        return relevant_found / k


    def mrr(self, retrieved_chunks, relevant_sections):

        for i, r in enumerate(retrieved_chunks):
            if self.is_relevant(r["section_title"], relevant_sections):
                return 1 / (i + 1)

        return 0


    def ndcg(self, retrieved_chunks, relevant_sections, k=5):

        dcg = 0

        for i, r in enumerate(retrieved_chunks[:k]):

            if self.is_relevant(r["section_title"], relevant_sections):
                dcg += 1 / math.log2(i + 2)

        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_sections), k)))

        if idcg == 0:
            return 0

        return dcg / idcg