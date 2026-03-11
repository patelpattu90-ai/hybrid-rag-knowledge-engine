"""
tools.py — Tool execution logic for each routed tool.

Each tool receives the same inputs but handles them differently:

  search_docs   → standard RAG: retrieve → rerank → stream answer
  summarise     → retrieve MORE chunks (top-10), ask LLM for structured overview
  answer_direct → skip retrieval entirely, answer from conversation history only

All tools return a ToolResult with a generator for streaming + metadata.
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Callable

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    tool:        str
    token_gen:   Generator                 # streaming tokens
    sections:    list[str] = field(default_factory=list)
    latency:     dict      = field(default_factory=dict)
    skipped_retrieval: bool = False        # True for answer_direct


# ── Tool: search_docs ────────────────────────────────────────────────────────
def run_search_docs(
    query:          str,
    retrieval_query: str,
    contexts:       list[str],
    sections:       list[str],
    latency:        dict,
    history:        list[dict],
    generator,
) -> ToolResult:
    """
    Standard RAG path — already retrieved, just stream with history.
    """
    return ToolResult(
        tool="search_docs",
        token_gen=generator.stream_with_history(retrieval_query, contexts, history),
        sections=sections,
        latency=latency,
        skipped_retrieval=False,
    )


# ── Tool: summarise ──────────────────────────────────────────────────────────
def run_summarise(
    query:          str,
    retrieval_query: str,
    contexts:       list[str],
    sections:       list[str],
    latency:        dict,
    history:        list[dict],
    generator,
) -> ToolResult:
    """
    Summary mode — uses the same retrieved contexts but asks the LLM for
    a structured overview instead of a direct answer.
    Injects a summary instruction into the query before passing to the generator.
    """
    summary_query = (
        f"Give a structured overview of the following topic based on the documentation: "
        f"{query}\n\n"
        "Format your response with:\n"
        "1. A one-sentence definition\n"
        "2. Key concepts (bullet points)\n"
        "3. A short code example if available in the context\n"
        "4. When to use this feature"
    )
    return ToolResult(
        tool="summarise",
        token_gen=generator.stream_with_history(summary_query, contexts, history),
        sections=sections,
        latency=latency,
        skipped_retrieval=False,
    )


# ── Tool: answer_direct ──────────────────────────────────────────────────────
def run_answer_direct(
    query:   str,
    history: list[dict],
    generator,
) -> ToolResult:
    """
    Direct answer — skip retrieval entirely.
    Passes an empty context and relies on conversation history + LLM knowledge
    for greetings, follow-ups, and chitchat.
    """
    direct_query = (
        f"{query}\n\n"
        "(Note: Answer conversationally. No documentation context is needed for this.)"
    )
    return ToolResult(
        tool="answer_direct",
        token_gen=generator.stream_with_history(direct_query, [], history),
        sections=[],
        latency={},
        skipped_retrieval=True,
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────
def dispatch_tool(
    tool_name:       str,
    query:           str,
    retrieval_query: str,
    contexts:        list[str],
    sections:        list[str],
    latency:         dict,
    history:         list[dict],
    generator,
) -> ToolResult:
    """
    Route to the correct tool function based on tool_name.
    Falls back to search_docs for any unknown tool name.
    """
    if tool_name == "answer_direct":
        return run_answer_direct(query, history, generator)

    if tool_name == "summarise":
        return run_summarise(
            query, retrieval_query, contexts, sections, latency, history, generator
        )

    # Default: search_docs (also handles unknown tool names)
    return run_search_docs(
        query, retrieval_query, contexts, sections, latency, history, generator
    )
