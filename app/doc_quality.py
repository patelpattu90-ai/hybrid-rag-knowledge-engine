"""
doc_quality.py — Document Quality Scoring

Problem this solves:
  Not all retrieved chunks are equally good. Some chunks are:
  - Too short to be useful (stub sections, headings only)
  - Too noisy (lots of markdown syntax, code-only, or navigation text)
  - Duplicates or near-duplicates of other chunks
  - Off-topic despite passing retrieval (false positives)

Solution — Quality Scoring:
  Score each chunk on 3 dimensions, combine into a final quality score,
  and use scores to filter or re-rank chunks AFTER retrieval but BEFORE
  compression and generation.

Three scoring dimensions (all regex/heuristic — no LLM call needed):

  1. LENGTH SCORE       — penalise very short or very long chunks
                         (too short = stub, too long = unfocused)

  2. CONTENT DENSITY    — ratio of alphanumeric chars to total chars
                         (low density = mostly markdown/whitespace noise)

  3. INFORMATION SCORE  — presence of informative patterns:
                         sentences, code examples, technical terms

Final quality score = weighted average of the three dimensions (0.0 – 1.0)
Chunks below MIN_QUALITY_SCORE are flagged (not dropped — flagged for UI display)
"""

import re
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tunable thresholds
_MIN_QUALITY_SCORE  = 0.35   # chunks below this are flagged as low quality
_IDEAL_CHUNK_CHARS  = 600    # sweet spot for chunk length
_MIN_CHUNK_CHARS    = 80     # too short to be useful
_MAX_CHUNK_CHARS    = 2000   # too long = unfocused

# Technical term patterns that indicate informative content
_TECHNICAL_PATTERNS = [
    r"\bdef\s+\w+\(",          # Python function definitions
    r"\bclass\s+\w+",          # class definitions
    r"@\w+",                   # decorators
    r"\b(GET|POST|PUT|DELETE|PATCH)\b",  # HTTP methods
    r"```",                    # code blocks
    r"\b(router|endpoint|middleware|dependency|schema|model|request|response)\b",
    r"\bFastAPI\b",
    r"\bPydantic\b",
    r"http[s]?://",            # URLs
    r"\bparameter[s]?\b",
    r"\bpath\s+operation",
]


@dataclass
class ChunkQuality:
    chunk_idx:       int
    length_score:    float
    density_score:   float
    info_score:      float
    final_score:     float
    is_low_quality:  bool
    char_count:      int

    def badge(self) -> str:
        if self.is_low_quality:
            return f"⚠️ Chunk {self.chunk_idx + 1}: low quality ({self.final_score:.2f})"
        return f"✅ Chunk {self.chunk_idx + 1}: quality {self.final_score:.2f}"


@dataclass
class QualityReport:
    chunks_scored:   int
    low_quality:     int
    avg_score:       float
    scores:          list[ChunkQuality] = field(default_factory=list)

    def summary_md(self) -> str:
        if not self.scores:
            return ""
        high = sum(1 for s in self.scores if s.final_score >= 0.6)
        mid  = sum(1 for s in self.scores if 0.35 <= s.final_score < 0.6)
        low  = self.low_quality
        return (
            f"📋 **Doc Quality:** {self.chunks_scored} chunks scored  ·  "
            f"🟢 {high} high  ·  🟡 {mid} medium  ·  🔴 {low} low  ·  "
            f"Avg: {self.avg_score:.2f}"
        )


class DocQualityScorer:
    """
    Scores retrieved chunks on length, content density, and information richness.
    Pure heuristic — zero LLM calls, runs in microseconds.
    """

    def __init__(self):
        self._tech_patterns = [re.compile(p, re.IGNORECASE) for p in _TECHNICAL_PATTERNS]

    # ── Dimension 1: Length score ─────────────────────────────────────────────
    def _length_score(self, text: str) -> float:
        n = len(text)
        if n < _MIN_CHUNK_CHARS:
            return 0.1
        if n > _MAX_CHUNK_CHARS:
            # Penalise proportionally for exceeding max
            return max(0.3, 1.0 - (n - _MAX_CHUNK_CHARS) / _MAX_CHUNK_CHARS)
        # Bell curve peaking at _IDEAL_CHUNK_CHARS
        distance = abs(n - _IDEAL_CHUNK_CHARS) / _IDEAL_CHUNK_CHARS
        return max(0.4, 1.0 - distance * 0.5)

    # ── Dimension 2: Content density ─────────────────────────────────────────
    def _density_score(self, text: str) -> float:
        if not text:
            return 0.0
        alnum = sum(1 for c in text if c.isalnum() or c == ' ')
        ratio = alnum / len(text)
        # Good docs: 60-85% alphanumeric+space ratio
        if ratio >= 0.6:
            return 1.0
        if ratio >= 0.4:
            return 0.7
        if ratio >= 0.2:
            return 0.4
        return 0.1   # mostly symbols/markdown

    # ── Dimension 3: Information score ───────────────────────────────────────
    def _info_score(self, text: str) -> float:
        # Count sentence-like structures
        sentences = len(re.findall(r'[.!?]\s+[A-Z]', text))
        sentence_score = min(1.0, sentences / 5)   # 5+ sentences = full score

        # Count technical patterns
        tech_hits = sum(1 for p in self._tech_patterns if p.search(text))
        tech_score = min(1.0, tech_hits / 3)       # 3+ patterns = full score

        return (sentence_score * 0.5) + (tech_score * 0.5)

    # ── Main scorer ───────────────────────────────────────────────────────────
    def score_chunk(self, chunk: str, idx: int) -> ChunkQuality:
        length_score  = self._length_score(chunk)
        density_score = self._density_score(chunk)
        info_score    = self._info_score(chunk)

        # Weighted average: length 30%, density 30%, info 40%
        final = (length_score * 0.3) + (density_score * 0.3) + (info_score * 0.4)
        final = round(final, 3)

        return ChunkQuality(
            chunk_idx=idx,
            length_score=length_score,
            density_score=density_score,
            info_score=info_score,
            final_score=final,
            is_low_quality=final < _MIN_QUALITY_SCORE,
            char_count=len(chunk),
        )

    def score_all(self, chunks: list[str]) -> QualityReport:
        """Score all chunks and return a QualityReport."""
        scores      = [self.score_chunk(c, i) for i, c in enumerate(chunks)]
        low_quality = sum(1 for s in scores if s.is_low_quality)
        avg_score   = round(sum(s.final_score for s in scores) / len(scores), 3) if scores else 0.0

        return QualityReport(
            chunks_scored=len(scores),
            low_quality=low_quality,
            avg_score=avg_score,
            scores=scores,
        )

    def filter_low_quality(
        self,
        chunks:  list[str],
        report:  QualityReport,
    ) -> list[str]:
        """
        Return only chunks that pass the quality threshold.
        If filtering would remove everything, return all chunks (safe fallback).
        """
        filtered = [
            c for c, s in zip(chunks, report.scores)
            if not s.is_low_quality
        ]
        if not filtered:
            logger.warning("All chunks failed quality filter — returning originals")
            return chunks
        return filtered
