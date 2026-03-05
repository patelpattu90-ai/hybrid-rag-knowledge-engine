import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#0f1117'
matplotlib.rcParams['axes.facecolor'] = '#1a1d2e'
matplotlib.rcParams['axes.edgecolor'] = '#2d3154'
matplotlib.rcParams['text.color'] = '#e2e8f0'
matplotlib.rcParams['axes.labelcolor'] = '#94a3b8'
matplotlib.rcParams['xtick.color'] = '#94a3b8'
matplotlib.rcParams['ytick.color'] = '#94a3b8'

from app.eval_dataset import eval_queries
from app.evaluator import RetrievalEvaluator, build_eval_chart
from app.ingestion import load_documents
from app.chunker import IntelligentChunker
from app.embeddings import EmbeddingIndexer
from app.bm25_retriever import BM25Retriever
from app.hybrid_search import HybridSearch
from app.reranker import CrossEncoderReranker
from app.generator import GroqGenerator
import re

def clean_title(title):
    title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
    title = re.sub(r'\{[^}]+\}', '', title)
    title = title.lstrip('#').strip()
    return title

# ---------------------------
# Load and Prepare Documents
# ---------------------------
docs = load_documents()
chunker = IntelligentChunker()
chunks = chunker.chunk_documents(docs)
texts = [c["text"] for c in chunks]

embedding_indexer = EmbeddingIndexer()
embedding_indexer.build_index(chunks)
bm25 = BM25Retriever(chunks)
hybrid = HybridSearch(embedding_indexer, bm25)
reranker = CrossEncoderReranker()
generator = GroqGenerator()

# ---------------------------
# Pipeline
# ---------------------------
def rag_pipeline(query):
    results, latency = hybrid.search(query)
    results = reranker.rerank(query, results)
    contexts = [r["text"] for r in results[:5]]
    answer = generator.generate(query, contexts)
    sections = [clean_title(r["section_title"]) for r in results[:5]]

    # Find matching eval entry for this query
    evaluator = RetrievalEvaluator()
    matched = next((e for e in eval_queries if e["query"].lower() == query.lower()), None)

    if matched:
        relevant_sections = matched["relevant_sections"]
    else:
        # Fallback: use the top retrieved section as pseudo ground truth
        relevant_sections = [sections[0]] if sections else []

    precision = evaluator.precision_at_k(results, relevant_sections)
    mrr_score = evaluator.mrr(results, relevant_sections)
    ndcg_score = evaluator.ndcg(results, relevant_sections)

    plot = build_eval_chart(precision, mrr_score, ndcg_score)

    return answer, sections, latency, plot

# ---------------------------
# Custom CSS
# ---------------------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #080b14 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header */
#header-md h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: -0.5px !important;
}

#subheader-md p {
    color: #64748b !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.3px !important;
    margin-bottom: 2rem !important;
}

/* Panels */
.panel {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* Labels */
label span, .label-wrap span {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

/* Textbox inputs */
textarea, input[type="text"] {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* JSON output */
.json-holder {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
}

/* Buttons */
#submit-btn button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
}

#submit-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99, 102, 241, 0.5) !important;
}

#clear-btn button {
    background: transparent !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    color: #475569 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#clear-btn button:hover {
    border-color: #475569 !important;
    color: #94a3b8 !important;
}

/* Plot */
.plot-container {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 16px !important;
}

/* Divider spacing */
.gap { gap: 1.2rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0d16; }
::-webkit-scrollbar-thumb { background: #1e2235; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
"""

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(title="Hybrid RAG Knowledge Engine", css=css, theme=gr.themes.Base()) as demo:

    with gr.Column():
        gr.Markdown("# ⚡ Hybrid RAG Knowledge Engine", elem_id="header-md")
        gr.Markdown("Semantic + BM25 retrieval · Cross-encoder reranking · LLM generation", elem_id="subheader-md")

    with gr.Row(equal_height=True):
        query = gr.Textbox(
            label="Ask a Question",
            placeholder="e.g. How does dependency injection work in FastAPI?",
            lines=4,
            elem_classes=["panel"]
        )
        answer = gr.Textbox(
            label="Generated Answer",
            lines=4,
            interactive=False,
            elem_classes=["panel"]
        )

    with gr.Row(equal_height=True):
        retrieved_sections = gr.JSON(label="Top Retrieved Sections", elem_classes=["panel"])
        latency = gr.JSON(label="Retrieval Latency (ms)", elem_classes=["panel"])

    chart = gr.Plot(label="Evaluation Metrics", elem_classes=["panel"])

    with gr.Row():
        clear = gr.Button("Clear", elem_id="clear-btn")
        submit = gr.Button("Submit →", elem_id="submit-btn")

    submit.click(rag_pipeline, inputs=query, outputs=[answer, retrieved_sections, latency, chart])
    clear.click(lambda: ("", [], {}, None), outputs=[answer, retrieved_sections, latency, chart])

if __name__ == "__main__":
    demo.launch()