"""
query_rewriter.py — Rewrite vague or ambiguous queries before retrieval.

Why this matters:
  "auth?"              → "How do I implement authentication in FastAPI?"
  "middleware"         → "How does middleware work in FastAPI and how do I add it?"
  "how fast is it?"    → "What is the performance and speed of FastAPI?"
  "deps"               → "How does dependency injection work in FastAPI?"

The rewriter calls Groq with a lightweight prompt.
If the LLM call fails for any reason, the ORIGINAL query is returned unchanged
so the pipeline never breaks.
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class RewriteResult:
    original:  str
    rewritten: str
    was_rewritten: bool   # False if fallback to original was used

    def display(self) -> str:
        """One-line string for the UI info panel."""
        if self.was_rewritten and self.original.strip().lower() != self.rewritten.strip().lower():
            return f"✏️ **Rewritten:** {self.rewritten}\n_Original: {self.original}_"
        return f"✅ **Query:** {self.rewritten}"


# ── Rewriter class ───────────────────────────────────────────────────────────
class QueryRewriter:

    # System prompt — tight and explicit so the LLM doesn't over-rewrite
    _SYSTEM_PROMPT = """You are a query rewriting assistant for a FastAPI documentation search engine.

Your job: rewrite the user's query into a clear, specific, self-contained question that will retrieve the best results from FastAPI documentation.

Rules:
1. Expand abbreviations (e.g. "auth" → "authentication", "deps" → "dependency injection")
2. Make vague queries specific (e.g. "how fast?" → "What is the performance of FastAPI?")
3. Add "in FastAPI" context if it is missing and the query is clearly about FastAPI
4. If the query is already clear and specific, return it UNCHANGED
5. Never add information that isn't implied by the original query
6. Return ONLY the rewritten query — no explanation, no prefix, no quotes"""

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"   # same model as generator

    def rewrite(self, query: str) -> RewriteResult:
        """
        Rewrite query. Always returns a RewriteResult.
        Falls back to original query if the LLM call fails.
        """
        query = query.strip()

        # Skip rewriting for empty or very long queries
        if not query:
            return RewriteResult(original=query, rewritten=query, was_rewritten=False)
        if len(query) > 500:
            return RewriteResult(original=query, rewritten=query, was_rewritten=False)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": query},
                ],
                temperature=0.0,   # deterministic — we want consistent rewrites
                max_tokens=120,    # a rewritten query should never be long
            )

            rewritten = response.choices[0].message.content.strip()

            # Sanity checks — fall back if output looks wrong
            if not rewritten:
                raise ValueError("Empty response from rewriter")
            if len(rewritten) > 300:
                raise ValueError("Rewritten query suspiciously long")
            # If LLM returned something like "Rewritten query: ..." strip the prefix
            for prefix in ("rewritten query:", "query:", "answer:"):
                if rewritten.lower().startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()

            was_rewritten = rewritten.lower() != query.lower()
            return RewriteResult(original=query, rewritten=rewritten, was_rewritten=was_rewritten)

        except Exception as e:
            logger.warning(f"QueryRewriter failed, using original query. Error: {e}")
            return RewriteResult(original=query, rewritten=query, was_rewritten=False)
