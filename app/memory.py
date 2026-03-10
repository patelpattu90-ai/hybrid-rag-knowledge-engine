"""
memory.py — Conversation memory with sliding window for multi-turn chat.

Why this matters:
  User: "What is FastAPI?"
  User: "How do I install it?"   ← "it" needs context from previous turn
  User: "Show me an example"     ← still needs to know we're talking about FastAPI

The memory stores the last N turns and injects them into the LLM prompt
so it can resolve pronouns, follow-ups, and topic continuations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal


# ── Single conversation turn ──────────────────────────────────────────────────
@dataclass
class Turn:
    role:    Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ── Conversation memory ───────────────────────────────────────────────────────
class ConversationMemory:
    """
    Sliding window conversation memory.

    - Stores up to `max_turns` most recent turns (user + assistant = 2 turns per exchange)
    - Older turns are automatically dropped when the window is full
    - Each Gradio session gets its own instance via gr.State()
    """

    def __init__(self, max_turns: int = 6):
        """
        max_turns=6 means the last 3 full exchanges (3 user + 3 assistant).
        This keeps the injected history concise while preserving enough context.
        """
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    # ── Add turns ─────────────────────────────────────────────────────────────
    def add_user(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._turns.append(Turn(role="assistant", content=content))

    # ── Retrieve ──────────────────────────────────────────────────────────────
    def get_history(self) -> list[dict]:
        """
        Returns turns as list of {"role": ..., "content": ...} dicts.
        Ready to be injected directly into Groq messages array.
        """
        return [t.to_dict() for t in self._turns]

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def turn_count(self) -> int:
        return len(self._turns)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def clear(self) -> None:
        self._turns.clear()

    # ── Display for UI ────────────────────────────────────────────────────────
    def display(self) -> str:
        """
        One-line status string for the UI memory panel.
        """
        if self.is_empty():
            return "💬 **Memory:** No history yet"
        exchanges = len(self._turns) // 2
        return (
            f"💬 **Memory:** {exchanges} exchange(s) stored  ·  "
            f"{self.turn_count()} / {self.max_turns} turns  ·  "
            f"Window: last {self.max_turns // 2} Q&A pairs"
        )
