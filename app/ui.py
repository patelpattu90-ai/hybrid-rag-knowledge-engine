import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#0f1117'
matplotlib.rcParams['axes.facecolor']   = '#1a1d2e'
matplotlib.rcParams['axes.edgecolor']   = '#2d3154'
matplotlib.rcParams['text.color']       = '#e2e8f0'
matplotlib.rcParams['axes.labelcolor']  = '#94a3b8'
matplotlib.rcParams['xtick.color']      = '#94a3b8'
matplotlib.rcParams['ytick.color']      = '#94a3b8'

from app.eval_dataset    import eval_queries
from app.evaluator       import RetrievalEvaluator, build_eval_chart
from app.ingestion       import load_documents
from app.chunker         import IntelligentChunker
from app.embeddings      import EmbeddingIndexer
from app.bm25_retriever  import BM25Retriever
from app.hybrid_search   import HybridSearch
from app.reranker        import CrossEncoderReranker
from app.generator       import GroqGenerator
from app.cache           import query_cache
from app.query_rewriter  import QueryRewriter
from app.memory          import ConversationMemory
from app.tool_router     import ToolRouter
from app.tools           import dispatch_tool
from app.guardrails      import Guardrails
from app.security        import input_sanitiser, pii_scrubber  # ← NEW (Day 6)
import re

def clean_title(title):
    title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
    title = re.sub(r'\{[^}]+\}', '', title)
    title = title.lstrip('#').strip()
    return title

# ---------------------------
# Load and Prepare Documents
# ---------------------------
docs              = load_documents()
chunker           = IntelligentChunker()
chunks            = chunker.chunk_documents(docs)
texts             = [c["text"] for c in chunks]

embedding_indexer = EmbeddingIndexer()
embedding_indexer.build_index(chunks)
bm25              = BM25Retriever(chunks)
hybrid            = HybridSearch(embedding_indexer, bm25)
reranker          = CrossEncoderReranker()
generator         = GroqGenerator()
rewriter          = QueryRewriter()
tool_router       = ToolRouter()
guardrails        = Guardrails()
# input_sanitiser + pii_scrubber are module-level singletons

# ---------------------------
# Pipeline
# ---------------------------
def rag_pipeline(query, memory: ConversationMemory):
    """
    Gradio generator — yields 9 values on every streaming token:
      answer, sections, latency, chart, cache_md,
      rewrite_md, memory_md, tool_md, guard_md
    """

    # ── 0a. INPUT SANITISATION (NEW Day 6) ────────────────────────────────────
    sanitise_result = input_sanitiser.sanitise(query)
    clean_query     = sanitise_result.sanitised

    # ── 0b. GUARDRAILS (Day 5) ────────────────────────────────────────────────
    guard    = guardrails.check(clean_query)
    guard_md = _guard_badge(guard, sanitise_result)

    if guard.blocked:
        yield (
            guard.user_message(),
            [], {}, None,
            _cache_stats_md(),
            "", memory.display(), "", guard_md,
        )
        return

    # ── 1. Cache check ────────────────────────────────────────────────────────
    cached = query_cache.get(clean_query, top_k=5)
    if cached is not None:
        yield (
            cached["answer"],
            cached["sections"],
            cached["latency"],
            cached["chart"],
            _cache_stats_md(hit=True),
            cached["rewrite_md"],
            memory.display(),
            cached.get("tool_md", ""),
            guard_md,
        )
        return

    # ── 2. Query rewriting ────────────────────────────────────────────────────
    rewrite         = rewriter.rewrite(clean_query)
    retrieval_query = rewrite.rewritten
    rewrite_md      = rewrite.display()

    # ── 3. Tool routing ───────────────────────────────────────────────────────
    route   = tool_router.route(clean_query)
    tool_md = route.badge()

    # ── 4. Retrieval ──────────────────────────────────────────────────────────
    contexts = []
    sections = []
    latency  = {}
    chart    = None

    if route.tool != "answer_direct":
        top_k = 10 if route.tool == "summarise" else 5

        results, latency = hybrid.search(retrieval_query)
        results          = reranker.rerank(retrieval_query, results)
        raw_contexts     = [r["text"]                       for r in results[:top_k]]
        sections         = [clean_title(r["section_title"]) for r in results[:top_k]]

        # ── PII SCRUB: chunk text before sending to LLM (NEW Day 6) ──────────
        contexts = pii_scrubber.scrub_contexts(raw_contexts)

        evaluator = RetrievalEvaluator()
        matched   = next((e for e in eval_queries if e["query"].lower() == clean_query.lower()), None)
        relevant_sections = matched["relevant_sections"] if matched else (sections[:1] if sections else [])

        precision  = evaluator.precision_at_k(results, relevant_sections)
        mrr_score  = evaluator.mrr(results, relevant_sections)
        ndcg_score = evaluator.ndcg(results, relevant_sections)
        chart      = build_eval_chart(precision, mrr_score, ndcg_score)

    # ── 5. Add user turn to memory ────────────────────────────────────────────
    memory.add_user(clean_query)
    history = memory.get_history()

    # ── 6. Dispatch to correct tool + stream ──────────────────────────────────
    tool_result = dispatch_tool(
        tool_name=route.tool,
        query=clean_query,
        retrieval_query=retrieval_query,
        contexts=contexts,
        sections=sections,
        latency=latency,
        history=history,
        generator=generator,
    )

    answer = ""
    for token in tool_result.token_gen:
        answer += token
        yield (
            answer,
            tool_result.sections,
            tool_result.latency,
            chart,
            _cache_stats_md(hit=False),
            rewrite_md,
            memory.display(),
            tool_md,
            guard_md,
        )

    # ── 7. PII SCRUB: generated answer before showing to user (NEW Day 6) ────
    answer = pii_scrubber.scrub_answer(answer)

    # ── 8. Add scrubbed answer to memory ─────────────────────────────────────
    memory.add_assistant(answer)

    # ── 9. Cache ──────────────────────────────────────────────────────────────
    if memory.turn_count() <= 2 and route.tool != "answer_direct":
        query_cache.set(clean_query, {
            "answer":     answer,
            "sections":   tool_result.sections,
            "latency":    tool_result.latency,
            "chart":      chart,
            "rewrite_md": rewrite_md,
            "tool_md":    tool_md,
        }, top_k=5)

    yield (
        answer,
        tool_result.sections,
        tool_result.latency,
        chart,
        _cache_stats_md(hit=False),
        rewrite_md,
        memory.display(),
        tool_md,
        guard_md,
    )


# ---------------------------
# Helpers
# ---------------------------
def _guard_badge(guard, sanitise_result) -> str:
    parts = []
    if sanitise_result.was_modified:
        parts.append(f"🔐 **Security:** {len(sanitise_result.changes)} pattern(s) stripped")
    else:
        parts.append("🔐 **Security:** Input clean")

    if guard.blocked:
        icons = {"jailbreak": "🚨", "harmful": "🚨", "off_topic": "⚠️"}
        parts.append(f"{icons.get(guard.layer, '⚠️')} **Guardrails:** Blocked — {guard.layer}")
    else:
        parts.append("🛡️ **Guardrails:** Passed")

    return "  ·  ".join(parts)


def _cache_stats_md(hit: bool = False) -> str:
    s     = query_cache.stats()
    badge = "🟢 **CACHE HIT** — returned instantly" if hit else "🔵 Live query"
    return (
        f"{badge}\n\n"
        f"**Entries:** {s['entries']} / {s['max_size']}  ·  "
        f"**Hit rate:** {s['hit_rate_pct']}%  ·  "
        f"**Hits:** {s['hits']}  ·  **Misses:** {s['misses']}"
    )

def _clear_cache():
    query_cache.clear()
    return _cache_stats_md()


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

.panel {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

label span, .label-wrap span {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

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

.json-holder {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
}

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

#cache-stats {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.82rem !important;
    color: #64748b !important;
}

#rewrite-info {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-left: 3px solid #6366f1 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

#memory-info {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-left: 3px solid #0f766e !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

#tool-info {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-left: 3px solid #c05621 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

#guard-info {
    background: #0a0d16 !important;
    border: 1px solid #1e2235 !important;
    border-left: 3px solid #dc2626 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

.plot-container {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 16px !important;
}

.gap { gap: 1.2rem !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0d16; }
::-webkit-scrollbar-thumb { background: #1e2235; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
"""

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(title="Hybrid RAG Knowledge Engine", css=css, theme=gr.themes.Base()) as demo:

    session_memory = gr.State(lambda: ConversationMemory(max_turns=6))

    with gr.Column():
        gr.Markdown("# ⚡ Hybrid RAG Knowledge Engine", elem_id="header-md")
        gr.Markdown(
            "Semantic + BM25 · Cross-encoder reranking · LLM generation · "
            "Streaming · Cache · Query Rewriting · Memory · Tool Routing · "
            "Guardrails · Prompt Injection Defense · PII Scrubbing",
            elem_id="subheader-md"
        )

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

    # ── Info panels: security+guard · rewrite · memory · tool ─────────────────
    with gr.Row():
        guard_info   = gr.Markdown(
            value="🔐 **Security:** Input clean  ·  🛡️ **Guardrails:** Ready",
            elem_id="guard-info"
        )
        rewrite_info = gr.Markdown(value="", elem_id="rewrite-info")
        memory_info  = gr.Markdown(value="💬 **Memory:** No history yet", elem_id="memory-info")
        tool_info    = gr.Markdown(value="", elem_id="tool-info")

    with gr.Row(equal_height=True):
        retrieved_sections = gr.JSON(label="Top Retrieved Sections", elem_classes=["panel"])
        latency            = gr.JSON(label="Retrieval Latency (ms)",  elem_classes=["panel"])

    chart = gr.Plot(label="Evaluation Metrics", elem_classes=["panel"])

    cache_stats = gr.Markdown(value=_cache_stats_md(), elem_id="cache-stats")

    with gr.Row():
        clear_history_btn = gr.Button("Clear History", elem_id="clear-btn")
        clear_cache_btn   = gr.Button("Clear Cache",   elem_id="clear-btn")
        clear_btn         = gr.Button("Clear Output",  elem_id="clear-btn")
        submit            = gr.Button("Submit →",      elem_id="submit-btn")

    _outputs = [
        answer, retrieved_sections, latency, chart,
        cache_stats, rewrite_info, memory_info, tool_info, guard_info,
    ]

    submit.click(fn=rag_pipeline, inputs=[query, session_memory], outputs=_outputs)
    query.submit(fn=rag_pipeline, inputs=[query, session_memory], outputs=_outputs)

    def clear_history(memory: ConversationMemory):
        memory.clear()
        return memory, "💬 **Memory:** Cleared"

    clear_history_btn.click(
        fn=clear_history,
        inputs=[session_memory],
        outputs=[session_memory, memory_info],
    )

    clear_btn.click(
        fn=lambda: ("", [], {}, None, _cache_stats_md(), "",
                    "💬 **Memory:** No history yet", "",
                    "🔐 **Security:** Input clean  ·  🛡️ **Guardrails:** Ready"),
        outputs=_outputs,
    )

    clear_cache_btn.click(fn=_clear_cache, outputs=[cache_stats])

if __name__ == "__main__":
    demo.launch()
