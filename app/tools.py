"""
tools.py — Tool execution logic for each routed tool.

Each tool receives the same inputs but handles them differently:

  search_docs   → standard RAG: retrieve → rerank → stream answer
  summarise     → retrieve MORE chunks (top-10), structured overview via
                  a dedicated system message (NOT by rewriting the query)
  answer_direct → skip retrieval entirely, answer from conversation history only

All tools return a ToolResult with a generator for streaming + metadata.

FIX (Day 6 patch): summarise no longer rewrites the query string.
Previously it replaced the query with formatting instructions, which caused
the LLM to receive "Give a structured overview..." as the question and fail
to match it against the context, returning "I could not find the answer."
Now it passes the ORIGINAL query + a separate summary system message.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Generator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    tool:             str
    token_gen:        Generator
    sections:         list[str] = field(default_factory=list)
    latency:          dict      = field(default_factory=dict)
    skipped_retrieval: bool     = False


# ── Tool: search_docs ─────────────────────────────────────────────────────────
def run_search_docs(
    query:           str,
    retrieval_query: str,
    contexts:        list[str],
    sections:        list[str],
    latency:         dict,
    history:         list[dict],
    generator,
) -> ToolResult:
    """Standard RAG path — retrieve then stream with history."""
    return ToolResult(
        tool="search_docs",
        token_gen=generator.stream_with_history(retrieval_query, contexts, history),
        sections=sections,
        latency=latency,
        skipped_retrieval=False,
    )


# ── Tool: summarise ───────────────────────────────────────────────────────────
def run_summarise(
    query:           str,
    retrieval_query: str,
    contexts:        list[str],
    sections:        list[str],
    latency:         dict,
    history:         list[dict],
    generator,
) -> ToolResult:
    """
    Summary mode — passes the ORIGINAL query to the LLM so context matching
    works correctly, but injects a structured-output instruction via a
    dedicated system message prepended to the history.

    The system message tells the LLM HOW to format the answer.
    The user message (via _context_block) carries the original question.
    This way the LLM matches context to the real question, not to formatting text.
    """
    _SUMMARISE_SYSTEM = {
        "role": "system",
        "content": (
            "You are a technical documentation assistant for FastAPI. "
            "Answer using ONLY the provided context chunks. "
            "Format your response as a structured overview:\n"
            "1. A one-sentence definition\n"
            "2. Key concepts (bullet points)\n"
            "3. A short code example if one is available in the context\n"
            "4. When to use this feature\n\n"
            "If the answer cannot be found in the context, say: "
            "'I could not find the answer in the documentation.'"
        ),
    }

    # Inject the summarise system message at the front of history
    summarise_history = [_SUMMARISE_SYSTEM] + history

    return ToolResult(
        tool="summarise",
        token_gen=generator.stream_with_history(
            retrieval_query,        # ← original query, NOT rewritten prompt
            contexts,
            summarise_history,      # ← history now carries the structure instruction
        ),
        sections=sections,
        latency=latency,
        skipped_retrieval=False,
    )


# ── Tool: answer_direct ───────────────────────────────────────────────────────
def run_answer_direct(
    query:   str,
    history: list[dict],
    generator,
) -> ToolResult:
    """
    Direct answer — skip retrieval entirely.
    Relies on conversation history + LLM knowledge for greetings and follow-ups.
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
    Route to the correct tool. Falls back to search_docs for unknown tool names.
    """
    if tool_name == "answer_direct":
        return run_answer_direct(query, history, generator)

    if tool_name == "summarise":
        return run_summarise(
            query, retrieval_query, contexts, sections, latency, history, generator
        )

    return run_search_docs(
        query, retrieval_query, contexts, sections, latency, history, generator
    )
