"""
ragas_eval.py — RAGAS-style evaluation pipeline

RAGAS (Retrieval Augmented Generation Assessment) measures 4 metrics:

  1. Faithfulness        — Does the answer contain ONLY claims supported by
                           the retrieved context? (hallucination detection)
                           Score: 0.0 – 1.0

  2. Answer Relevancy    — Is the answer relevant to the original question?
                           (measures if the LLM stayed on topic)
                           Score: 0.0 – 1.0

  3. Context Precision   — Are the retrieved chunks actually useful?
                           (measures retrieval quality)
                           Score: 0.0 – 1.0

  4. Context Recall      — Did retrieval find everything needed to answer?
                           (measures coverage)
                           Score: 0.0 – 1.0

All 4 metrics are computed via LLM-as-judge using a single Groq call each.
Returns structured scores + a radar chart for the UI.

Usage:
    evaluator = RAGASEvaluator()
    result    = evaluator.evaluate(query, answer, contexts)
    chart     = build_ragas_chart(result)
"""

import os
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RAGASResult:
    faithfulness:     float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall:   float = 0.0
    error:            str   = ""   # non-empty if evaluation failed

    @property
    def overall(self) -> float:
        """Simple average of all four scores."""
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_precision + self.context_recall) / 4, 3
        )

    def as_dict(self) -> dict:
        return {
            "faithfulness":      round(self.faithfulness, 3),
            "answer_relevancy":  round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall":    round(self.context_recall, 3),
            "overall":           self.overall,
        }

    def summary_md(self) -> str:
        if self.error:
            return f"⚠️ **RAGAS:** Evaluation failed — {self.error}"
        bars = {
            "Faithfulness":      self.faithfulness,
            "Answer Relevancy":  self.answer_relevancy,
            "Context Precision": self.context_precision,
            "Context Recall":    self.context_recall,
        }
        lines = ["**📊 RAGAS Scores**"]
        for name, score in bars.items():
            filled = int(score * 10)
            bar    = "█" * filled + "░" * (10 - filled)
            lines.append(f"`{bar}` **{name}:** {score:.2f}")
        lines.append(f"\n**Overall:** {self.overall:.2f}")
        return "\n\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are an evaluation assistant. 

Given a QUESTION, CONTEXT, and ANSWER, assess whether every claim in the ANSWER
is directly supported by the CONTEXT.

Score from 0.0 to 1.0:
- 1.0 = every claim in the answer is supported by the context
- 0.5 = some claims supported, some are hallucinated or external
- 0.0 = the answer is mostly hallucinated or contradicts the context

Respond ONLY with a JSON object like: {{"score": 0.85}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}"""


_ANSWER_RELEVANCY_PROMPT = """You are an evaluation assistant.

Given a QUESTION and ANSWER, assess how relevant and directly responsive
the answer is to the question.

Score from 0.0 to 1.0:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer is partially relevant or includes significant off-topic content
- 0.0 = answer does not address the question at all

Respond ONLY with a JSON object like: {{"score": 0.75}}

QUESTION: {question}

ANSWER: {answer}"""


_CONTEXT_PRECISION_PROMPT = """You are an evaluation assistant.

Given a QUESTION and a list of CONTEXT CHUNKS, assess what fraction of the
retrieved chunks are actually useful for answering the question.

Score from 0.0 to 1.0:
- 1.0 = all retrieved chunks are relevant to answering the question
- 0.5 = about half the chunks are relevant
- 0.0 = none of the chunks are useful for answering the question

Respond ONLY with a JSON object like: {{"score": 0.60}}

QUESTION: {question}

CONTEXT CHUNKS:
{context}"""


_CONTEXT_RECALL_PROMPT = """You are an evaluation assistant.

Given a QUESTION, CONTEXT, and ANSWER, assess whether the retrieved context
contains enough information to fully support the answer.

Score from 0.0 to 1.0:
- 1.0 = the context fully covers everything needed to answer the question
- 0.5 = the context partially covers the answer
- 0.0 = the context is missing most information needed to answer

Respond ONLY with a JSON object like: {{"score": 0.90}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}"""


# ── Evaluator ─────────────────────────────────────────────────────────────────
class RAGASEvaluator:
    """
    LLM-as-judge RAGAS evaluator.
    Makes 4 Groq calls (one per metric) and returns RAGASResult.
    All calls use temperature=0 for deterministic scoring.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

    def _score(self, prompt: str) -> float:
        """
        Make one LLM call and parse the score from JSON response.
        Returns 0.0 on any failure — never raises.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            raw  = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw  = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))   # clamp to [0, 1]
        except Exception as e:
            logger.warning(f"RAGAS score parse failed: {e} — returning 0.0")
            return 0.0

    def evaluate(
        self,
        query:    str,
        answer:   str,
        contexts: list[str],
    ) -> RAGASResult:
        """
        Run all 4 RAGAS metrics. Returns RAGASResult.
        Skips evaluation if answer or contexts are empty.
        """
        if not answer or not contexts:
            return RAGASResult(error="No answer or contexts to evaluate")

        context_text = "\n\n---\n\n".join(contexts[:5])   # cap at 5 chunks

        faithfulness = self._score(
            _FAITHFULNESS_PROMPT.format(
                question=query,
                context=context_text,
                answer=answer,
            )
        )

        answer_relevancy = self._score(
            _ANSWER_RELEVANCY_PROMPT.format(
                question=query,
                answer=answer,
            )
        )

        context_precision = self._score(
            _CONTEXT_PRECISION_PROMPT.format(
                question=query,
                context=context_text,
            )
        )

        context_recall = self._score(
            _CONTEXT_RECALL_PROMPT.format(
                question=query,
                context=context_text,
                answer=answer,
            )
        )

        return RAGASResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_ragas_chart(result: RAGASResult):
    """
    Build a horizontal bar chart showing all 4 RAGAS scores + overall.
    Returns a matplotlib Figure for gr.Plot().
    """
    if result.error:
        return None

    labels = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
    ]
    scores = [
        result.faithfulness,
        result.answer_relevancy,
        result.context_precision,
        result.context_recall,
    ]

    # Colour each bar by score: red < 0.5, amber 0.5–0.75, green > 0.75
    def bar_colour(s):
        if s >= 0.75:
            return "#22c55e"   # green
        if s >= 0.5:
            return "#f59e0b"   # amber
        return "#ef4444"       # red

    colours = [bar_colour(s) for s in scores]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, scores, color=colours, height=0.5)

    # Score labels inside bars
    for bar, score in zip(bars, scores):
        ax.text(
            max(score - 0.05, 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center", ha="right",
            color="white", fontsize=10, fontweight="bold"
        )

    # Overall score line
    overall = result.overall
    ax.axvline(overall, color="#a78bfa", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(overall + 0.01, -0.6, f"Overall: {overall:.2f}",
            color="#a78bfa", fontsize=9)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title("RAGAS Evaluation", fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
