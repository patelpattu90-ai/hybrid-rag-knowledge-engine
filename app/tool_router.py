"""
tool_router.py — Classify query intent and route to the correct tool.

Three tools:
  search_docs     → RAG retrieval pipeline (default for factual questions)
  summarise       → Broad topic overview using retrieved docs
  answer_direct   → Answer from memory/context alone, no retrieval needed

The router makes ONE fast Groq call (temperature=0, max_tokens=10)
and returns the tool name as a plain string.
Fallback: if the call fails or returns garbage → "search_docs"
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

# ── Valid tool names ──────────────────────────────────────────────────────────
TOOL_SEARCH    = "search_docs"
TOOL_SUMMARISE = "summarise"
TOOL_DIRECT    = "answer_direct"
VALID_TOOLS    = {TOOL_SEARCH, TOOL_SUMMARISE, TOOL_DIRECT}
FALLBACK_TOOL  = TOOL_SEARCH


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RouteResult:
    tool:       str    # one of VALID_TOOLS
    confidence: str    # "high" | "low" (low = fallback was used)
    reason:     str    # short explanation for the UI badge

    def badge(self) -> str:
        icons = {
            TOOL_SEARCH:    "🔍",
            TOOL_SUMMARISE: "📋",
            TOOL_DIRECT:    "💬",
        }
        labels = {
            TOOL_SEARCH:    "Search Docs",
            TOOL_SUMMARISE: "Summarise",
            TOOL_DIRECT:    "Direct Answer",
        }
        icon  = icons.get(self.tool, "🔍")
        label = labels.get(self.tool, self.tool)
        conf  = "" if self.confidence == "high" else " _(fallback)_"
        return f"{icon} **Tool:** {label}{conf}  ·  _{self.reason}_"


# ── Router ────────────────────────────────────────────────────────────────────
class ToolRouter:
    """
    Zero-shot intent classifier.
    Makes a single cheap Groq call to pick the right tool.
    Falls back to search_docs silently on any error.
    """

    _SYSTEM = """You are a query intent classifier for a FastAPI documentation assistant.

Given a user query, output EXACTLY ONE of these three words — nothing else:

  search_docs    → the user wants a specific factual answer from docs
                   (e.g. "How do I add OAuth2?", "What is a path parameter?")

  summarise      → the user wants a broad overview or comparison of a topic
                   (e.g. "Explain middleware", "Overview of FastAPI security",
                    "What are all the ways to handle errors?")

  answer_direct  → the query is conversational, a greeting, a follow-up that
                   needs no new retrieval, or clearly off-topic
                   (e.g. "Thanks", "What did you just say?", "Hello",
                    "Can you repeat that?")

Output only one of: search_docs | summarise | answer_direct"""

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

    def route(self, query: str) -> RouteResult:
        """
        Classify query intent. Always returns a RouteResult.
        Falls back to search_docs if LLM call fails.
        """
        query = query.strip()

        # Hardcode obvious direct answers without calling the LLM
        _direct_triggers = {"hi", "hello", "thanks", "thank you", "ok", "okay", "bye"}
        if query.lower() in _direct_triggers:
            return RouteResult(
                tool=TOOL_DIRECT,
                confidence="high",
                reason="Greeting / chitchat detected locally"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user",   "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            raw  = response.choices[0].message.content.strip().lower()
            tool = raw.split()[0] if raw else ""
            tool = tool.strip(".,!?\"'")

            if tool not in VALID_TOOLS:
                logger.warning(f"ToolRouter got unexpected output '{raw}', falling back")
                return RouteResult(
                    tool=FALLBACK_TOOL,
                    confidence="low",
                    reason="Unexpected classifier output → fallback"
                )

            reasons = {
                TOOL_SEARCH:    "Specific factual question → retrieval",
                TOOL_SUMMARISE: "Broad topic → retrieval + summary mode",
                TOOL_DIRECT:    "Conversational → no retrieval needed",
            }
            return RouteResult(tool=tool, confidence="high", reason=reasons[tool])

        except Exception as e:
            logger.warning(f"ToolRouter LLM call failed: {e} — using fallback")
            return RouteResult(
                tool=FALLBACK_TOOL,
                confidence="low",
                reason="Router error → fallback to search"
            )
