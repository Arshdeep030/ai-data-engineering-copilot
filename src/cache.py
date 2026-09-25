from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests

        if total == 0:
            return 0.0

        return self.hits / total


class InMemoryCache:
    """
    Simple thread-safe in-memory cache.

    This is intentionally simple for Day 10.
    Later this can be replaced with Redis without
    changing the RAG pipeline itself.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._stats = CacheStats()
        self._lock = Lock()

    @staticmethod
    def normalize_key(question: str) -> str:
        """
        Normalize user questions so trivial differences
        such as leading/trailing whitespace and casing
        don't create separate cache entries.
        """
        return " ".join(question.strip().lower().split())

    def get(self, question: str) -> Optional[Any]:
        key = self.normalize_key(question)

        with self._lock:
            if key in self._cache:
                self._stats.hits += 1
                return self._cache[key]

            self._stats.misses += 1
            return None

    def set(self, question: str, value: Any) -> None:
        key = self.normalize_key(question)

        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "total_requests": self._stats.total_requests,
                "hit_rate": round(self._stats.hit_rate, 4),
                "entries": len(self._cache),
            }


cache = InMemoryCache()
