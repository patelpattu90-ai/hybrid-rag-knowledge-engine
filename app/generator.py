import os
from typing import Generator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqGenerator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    # ── Shared system prompt ─────────────────────────────────────────────────
    _SYSTEM = (
        "You are a technical documentation assistant for FastAPI. "
        "Answer using ONLY the provided context chunks. "
        "If the answer cannot be found in the context, say: "
        "'I could not find the answer in the documentation.' "
        "If the user refers to something from earlier in the conversation, "
        "use the conversation history to resolve it."
    )

    # ── Context block builder (used by history-aware method) ─────────────────
    def _context_block(self, query: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(contexts)
        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    # ── Original prompt builder (kept for backwards compat) ──────────────────
    def _build_prompt(self, query: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(contexts)
        return f"""You are a technical documentation assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the documentation."

Context:
{context_text}

Question:
{query}

Answer:
"""

    # ── Non-streaming — no history (original, kept intact) ───────────────────
    def generate(self, query: str, contexts: list[str]) -> str:
        """
        Blocking call. Returns complete answer string.
        Used by evaluator and anywhere streaming is not needed.
        """
        prompt = self._build_prompt(query, contexts)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    # ── Streaming — no history (Day 1, kept intact) ───────────────────────────
    def stream(self, query: str, contexts: list[str]) -> Generator[str, None, None]:
        """
        Streaming call without conversation history.
        Kept intact so existing call sites don't break.
        """
        prompt = self._build_prompt(query, contexts)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ── Streaming — WITH conversation history (NEW Day 3) ────────────────────
    def stream_with_history(
        self,
        query:    str,
        contexts: list[str],
        history:  list[dict],          # from ConversationMemory.get_history()
    ) -> Generator[str, None, None]:
        """
        Streaming call that injects conversation history into the messages array.

        Message structure sent to Groq:
          [system]                         ← FastAPI assistant instructions
          [user]   turn 1 from history     ← oldest remembered exchange
          [assistant] turn 1 from history
          ...up to max_turns history...
          [user]   context + current query ← current question with retrieved docs

        This lets the LLM resolve follow-ups like "how do I install IT?" where
        "it" refers to something mentioned in a previous turn.
        """
        current_user_msg = self._context_block(query, contexts)

        messages = (
            [{"role": "system", "content": self._SYSTEM}]
            + history                                         # prior turns
            + [{"role": "user", "content": current_user_msg}]
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
