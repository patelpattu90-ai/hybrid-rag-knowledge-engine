import math
import matplotlib.pyplot as plt


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


# ✅ Moved OUTSIDE the class — now importable as a module-level function
def build_eval_chart(precision, mrr, ndcg):
    metrics = ["Precision@5", "MRR", "nDCG@5"]
    values = [precision, mrr, ndcg]
    colors = ["#6366f1", "#8b5cf6", "#a78bfa"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(metrics, values, color=colors, width=0.4, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_title("RAG Retrieval Evaluation Metrics", fontsize=13, pad=14, color="#e2e8f0")
    ax.set_ylabel("Score", fontsize=10)
    ax.grid(axis='y', color='#2d3154', linewidth=0.8, zorder=0)
    ax.spines[:].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha='center', va='bottom', fontsize=11,
                color='#e2e8f0', fontweight='bold')
    fig.tight_layout()
    return fig