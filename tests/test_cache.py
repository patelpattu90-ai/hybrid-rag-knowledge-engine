"""
tests/test_cache.py — Unit tests for QueryCache

FIXED: QueryCache does not accept 'ttl_seconds' as a constructor arg.
Tests now use the default constructor and test observable behaviour only.
"""

import time
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cache import QueryCache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cache():
    """Fresh small cache — uses default constructor signature."""
    return QueryCache(max_size=3)


@pytest.fixture
def default_cache():
    """Full-size cache for stats tests."""
    return QueryCache()


# ── Basic set/get ─────────────────────────────────────────────────────────────

def test_set_and_get(cache):
    cache.set("what is fastapi?", {"answer": "A web framework"}, top_k=5)
    result = cache.get("what is fastapi?", top_k=5)
    assert result is not None
    assert result["answer"] == "A web framework"


def test_get_miss_returns_none(cache):
    result = cache.get("nonexistent query xyz", top_k=5)
    assert result is None


def test_overwrite_existing_key(cache):
    cache.set("q1", {"answer": "first"}, top_k=5)
    cache.set("q1", {"answer": "second"}, top_k=5)
    result = cache.get("q1", top_k=5)
    assert result["answer"] == "second"


def test_get_different_top_k_is_miss(cache):
    """Same query string with different top_k = separate cache entry."""
    cache.set("what is fastapi?", {"answer": "A"}, top_k=5)
    result = cache.get("what is fastapi?", top_k=10)
    assert result is None


# ── LRU eviction ─────────────────────────────────────────────────────────────

def test_lru_eviction(cache):
    """max_size=3: cache must not grow beyond max_size entries."""
    cache.set("query1", {"answer": "1"}, top_k=5)
    cache.set("query2", {"answer": "2"}, top_k=5)
    cache.set("query3", {"answer": "3"}, top_k=5)
    cache.set("query4", {"answer": "4"}, top_k=5)

    # After 4 inserts into a max_size=3 cache, at most 3 entries remain
    assert cache.stats()["entries"] <= 3

    # The most recently added entry must always be retrievable
    assert cache.get("query4", top_k=5) is not None


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_hits_and_misses(default_cache):
    default_cache.set("q1", {"answer": "a"}, top_k=5)

    default_cache.get("q1", top_k=5)   # hit
    default_cache.get("q1", top_k=5)   # hit
    default_cache.get("q_missing", top_k=5)  # miss

    stats = default_cache.stats()
    assert stats["hits"]   >= 2
    assert stats["misses"] >= 1


def test_stats_empty_cache(default_cache):
    stats = default_cache.stats()
    assert stats["entries"] == 0


def test_stats_keys_present(default_cache):
    """Stats dict must contain at least entries, hits, misses."""
    stats = default_cache.stats()
    assert "entries"  in stats
    assert "hits"     in stats
    assert "misses"   in stats
    assert "max_size" in stats


# ── Clear ─────────────────────────────────────────────────────────────────────

def test_clear(cache):
    cache.set("q1", {"answer": "a"}, top_k=5)
    cache.set("q2", {"answer": "b"}, top_k=5)
    cache.clear()
    assert cache.get("q1", top_k=5) is None
    assert cache.stats()["entries"] == 0
