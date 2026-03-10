"""
cache.py — In-memory LRU cache for repeated query results.

Features:
- Exact-match cache keyed on SHA-256 of (query, top_k)
- TTL expiry  (default 1 hour)
- Max-size LRU eviction  (default 256 entries)
- Hit / miss stats for the UI panel
"""

import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


DEFAULT_MAX_SIZE   = 256
DEFAULT_TTL_SECONDS = 3600   # 1 hour


@dataclass
class _CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float = DEFAULT_TTL_SECONDS

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class QueryCache:

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE, ttl: float = DEFAULT_TTL_SECONDS):
        self.max_size = max_size
        self.ttl      = ttl
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits   = 0
        self._misses = 0

    # ── Key ─────────────────────────────────────────────────────────────────
    @staticmethod
    def make_key(query: str, top_k: int = 5) -> str:
        raw = f"{query.strip().lower()}|topk={top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Get ──────────────────────────────────────────────────────────────────
    def get(self, query: str, top_k: int = 5) -> Optional[Any]:
        key   = self.make_key(query, top_k)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._store[key]
            self._misses += 1
            return None

        self._store.move_to_end(key)   # mark as recently used
        self._hits += 1
        return entry.value

    # ── Set ──────────────────────────────────────────────────────────────────
    def set(self, query: str, value: Any, top_k: int = 5) -> None:
        key = self.make_key(query, top_k)

        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value=value, ttl=self.ttl)
            return

        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)   # evict least-recently-used

        self._store[key] = _CacheEntry(value=value, ttl=self.ttl)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def clear(self) -> None:
        self._store.clear()
        self._hits   = 0
        self._misses = 0

    def stats(self) -> dict:
        total    = self._hits + self._misses
        hit_rate = round(self._hits / total * 100, 1) if total > 0 else 0.0
        return {
            "entries":      len(self._store),
            "max_size":     self.max_size,
            "hits":         self._hits,
            "misses":       self._misses,
            "hit_rate_pct": hit_rate,
            "ttl_seconds":  self.ttl,
        }


# Singleton — import this everywhere
query_cache = QueryCache()
