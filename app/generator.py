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

    # ── Shared prompt builder ────────────────────────────────────────────────
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

    # ── Non-streaming (original — kept intact) ───────────────────────────────
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

    # ── Streaming (NEW for Day 1) ────────────────────────────────────────────
    def stream(self, query: str, contexts: list[str]) -> Generator[str, None, None]:
        """
        Streaming call. Yields token strings one at a time.

        Usage in Gradio:
            partial = ""
            for token in generator.stream(query, contexts):
                partial += token
                yield partial        # Gradio updates the textbox live
        """
        prompt = self._build_prompt(query, contexts)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,            # ← the only difference
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
