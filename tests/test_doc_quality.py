"""
tests/test_doc_quality.py — Unit tests for DocQualityScorer

Tests:
  - Short stub chunks score low
  - Rich technical chunks score high
  - All-symbol chunks score low density
  - filter_low_quality removes flagged chunks
  - filter_low_quality fallback: never returns empty list
  - QualityReport summary_md returns correct counts
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.doc_quality import DocQualityScorer, DocQualityScorer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return DocQualityScorer()


GOOD_CHUNK = """
Path parameters are declared in the path template using curly braces.
FastAPI will parse and validate them automatically.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

If the parameter is declared with a type annotation, FastAPI will perform
automatic data validation and conversion. This is one of the key features
that makes FastAPI powerful for building REST APIs.
"""

STUB_CHUNK = "See also"

NOISY_CHUNK = "---\n###\n---\n###\n**\n---"

MEDIUM_CHUNK = """
Query parameters are the set of key-value pairs that appear after the
question mark in a URL. They are optional by default in FastAPI.
"""


# ── Score chunk ───────────────────────────────────────────────────────────────

def test_good_chunk_scores_high(scorer):
    result = scorer.score_chunk(GOOD_CHUNK, idx=0)
    assert result.final_score >= 0.5
    assert result.is_low_quality is False


def test_stub_chunk_scores_low(scorer):
    result = scorer.score_chunk(STUB_CHUNK, idx=0)
    assert result.final_score < 0.5
    assert result.is_low_quality is True


def test_noisy_chunk_scores_low_density(scorer):
    result = scorer.score_chunk(NOISY_CHUNK, idx=0)
    assert result.density_score < 0.5


def test_medium_chunk_scored(scorer):
    result = scorer.score_chunk(MEDIUM_CHUNK, idx=0)
    assert 0.0 < result.final_score <= 1.0


# ── Score all ─────────────────────────────────────────────────────────────────

def test_score_all_returns_correct_count(scorer):
    chunks = [GOOD_CHUNK, STUB_CHUNK, MEDIUM_CHUNK]
    report = scorer.score_all(chunks)
    assert report.chunks_scored == 3
    assert len(report.scores) == 3


def test_score_all_avg_makes_sense(scorer):
    chunks = [GOOD_CHUNK, GOOD_CHUNK, GOOD_CHUNK]
    report = scorer.score_all(chunks)
    assert report.avg_score >= 0.5


def test_score_all_low_quality_count(scorer):
    chunks = [STUB_CHUNK, STUB_CHUNK, GOOD_CHUNK]
    report = scorer.score_all(chunks)
    assert report.low_quality == 2


# ── Filter ────────────────────────────────────────────────────────────────────

def test_filter_removes_low_quality(scorer):
    chunks = [GOOD_CHUNK, STUB_CHUNK, MEDIUM_CHUNK]
    report = scorer.score_all(chunks)
    filtered = scorer.filter_low_quality(chunks, report)
    assert STUB_CHUNK not in filtered
    assert GOOD_CHUNK in filtered


def test_filter_fallback_never_empty(scorer):
    """If all chunks are low quality, fallback returns all originals."""
    chunks = [STUB_CHUNK, NOISY_CHUNK]
    report = scorer.score_all(chunks)
    filtered = scorer.filter_low_quality(chunks, report)
    assert len(filtered) > 0


# ── Summary markdown ──────────────────────────────────────────────────────────

def test_summary_md_not_empty(scorer):
    chunks = [GOOD_CHUNK, STUB_CHUNK]
    report = scorer.score_all(chunks)
    md = report.summary_md()
    assert "Doc Quality" in md
    assert "chunks scored" in md
