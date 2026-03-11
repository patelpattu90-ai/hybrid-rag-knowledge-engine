"""
guardrails.py — Block off-topic, harmful, and jailbreak queries before
they reach the LLM or retrieval pipeline.

Three check layers (in order, fastest first):
  1. Jailbreak patterns  — regex, no LLM call needed
  2. Harmful keywords    — regex, no LLM call needed
  3. Topic classifier    — one fast Groq call (only if layers 1+2 pass)

Returns a GuardrailResult with blocked=True/False and a user-facing reason.
The pipeline checks this FIRST — before rewriting, routing, or retrieval.
"""

import os
import re
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class GuardrailResult:
    blocked: bool
    reason:  str          # internal reason (logged, not shown to user)
    layer:   str          # "jailbreak" | "harmful" | "off_topic" | "passed"

    def user_message(self) -> str:
        """Polite message shown in the UI — never reveals internal details."""
        messages = {
            "jailbreak": (
                "⚠️ **Request blocked** — This query contains patterns that "
                "cannot be processed. Please ask a question about FastAPI documentation."
            ),
            "harmful": (
                "⚠️ **Request blocked** — This query contains content that "
                "cannot be processed. Please ask a question about FastAPI documentation."
            ),
            "off_topic": (
                "⚠️ **Off-topic query** — This assistant is specialised for "
                "FastAPI documentation. Please ask about FastAPI, Python web APIs, "
                "routing, middleware, authentication, or related topics."
            ),
        }
        return messages.get(self.layer, "⚠️ **Request blocked.**")

    def badge(self) -> str:
        if not self.blocked:
            return "🛡️ **Guardrails:** Passed"
        icons = {"jailbreak": "🚨", "harmful": "🚨", "off_topic": "⚠️"}
        return f"{icons.get(self.layer, '⚠️')} **Guardrails:** Blocked — {self.layer}"


# ── Pattern lists ─────────────────────────────────────────────────────────────

# Jailbreak — attempts to override system instructions
_JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(a\s+)?(?!fastapi|python)",   # "you are now a [different persona]"
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(?!fastapi|python)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"unrestricted\s+mode",
    r"bypass\s+(your\s+)?(safety|filter|restriction|guardrail)",
    r"override\s+(your\s+)?(system|instruction|prompt)",
    r"\[system\]",
    r"<\|system\|>",
    r"<\|im_start\|>",
    r"prompt\s+injection",
    r"new\s+instructions?:",
    r"instructions?:\s*ignore",
]

# Harmful — dangerous or clearly malicious requests
_HARMFUL_PATTERNS = [
    r"\b(how\s+to\s+)?(make|build|create|synthesize)\s+(a\s+)?(bomb|weapon|explosive|malware|virus|ransomware)",
    r"\b(hack|exploit|attack|ddos|phish)\b.{0,30}\b(server|system|network|database|user)\b",
    r"\bself[- ]harm\b",
    r"\bsuicid",
    r"\b(child|minor).{0,20}(explicit|sexual|nude|naked)\b",
    r"\b(drug|narcotic).{0,20}(synthesize|make|produce|cook)\b",
    r"\bsteal\s+(password|credential|token|api.?key)\b",
]


# ── Guardrails class ──────────────────────────────────────────────────────────
class Guardrails:
    """
    Three-layer guard:
      Layer 1 — Jailbreak regex  (instant, no LLM)
      Layer 2 — Harmful regex    (instant, no LLM)
      Layer 3 — Off-topic LLM    (one cheap Groq call)
    """

    _TOPIC_SYSTEM = """You are a content classifier for a FastAPI documentation assistant.

Decide if the user's query is ON-TOPIC or OFF-TOPIC.

ON-TOPIC means the query is about:
- FastAPI framework, features, or usage
- Python web development, APIs, HTTP
- Pydantic, Starlette, Uvicorn, ASGI
- General programming questions related to building APIs
- Follow-up questions or greetings in the context of the above

OFF-TOPIC means the query is completely unrelated:
- Cooking, sports, weather, news, finance, geography, etc.
- Requests to generate creative fiction, poetry, songs
- Political or social opinions
- Anything clearly unrelated to software / APIs

Output EXACTLY one word: ON or OFF"""

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

        # Pre-compile all regex patterns for speed
        flags = re.IGNORECASE | re.DOTALL
        self._jailbreak_re = [re.compile(p, flags) for p in _JAILBREAK_PATTERNS]
        self._harmful_re   = [re.compile(p, flags) for p in _HARMFUL_PATTERNS]

    # ── Layer 1: Jailbreak ────────────────────────────────────────────────────
    def _check_jailbreak(self, query: str) -> GuardrailResult | None:
        for pattern in self._jailbreak_re:
            if pattern.search(query):
                logger.warning(f"Jailbreak pattern matched: {pattern.pattern[:40]}")
                return GuardrailResult(
                    blocked=True,
                    reason=f"Jailbreak pattern: {pattern.pattern[:40]}",
                    layer="jailbreak",
                )
        return None

    # ── Layer 2: Harmful ──────────────────────────────────────────────────────
    def _check_harmful(self, query: str) -> GuardrailResult | None:
        for pattern in self._harmful_re:
            if pattern.search(query):
                logger.warning(f"Harmful pattern matched: {pattern.pattern[:40]}")
                return GuardrailResult(
                    blocked=True,
                    reason=f"Harmful pattern: {pattern.pattern[:40]}",
                    layer="harmful",
                )
        return None

    # ── Layer 3: Off-topic LLM ────────────────────────────────────────────────
    def _check_topic(self, query: str) -> GuardrailResult | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._TOPIC_SYSTEM},
                    {"role": "user",   "content": query},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            verdict = response.choices[0].message.content.strip().upper()

            if verdict.startswith("OFF"):
                return GuardrailResult(
                    blocked=True,
                    reason="LLM classifier: off-topic query",
                    layer="off_topic",
                )
        except Exception as e:
            # If the LLM call fails, let it through — don't block valid queries
            logger.warning(f"Guardrail topic check failed: {e} — allowing query")

        return None

    # ── Main entry point ──────────────────────────────────────────────────────
    def check(self, query: str) -> GuardrailResult:
        """
        Run all three layers in order. Returns on first block.
        If all layers pass, returns GuardrailResult(blocked=False).
        """
        query = query.strip()

        # Layer 1 — jailbreak (no LLM, instant)
        result = self._check_jailbreak(query)
        if result:
            return result

        # Layer 2 — harmful (no LLM, instant)
        result = self._check_harmful(query)
        if result:
            return result

        # Layer 3 — off-topic (one LLM call)
        result = self._check_topic(query)
        if result:
            return result

        return GuardrailResult(blocked=False, reason="All checks passed", layer="passed")
