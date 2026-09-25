import sys
import unittest
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cache import InMemoryCache


class TestInMemoryCache(unittest.TestCase):

    def test_cache_miss(self):
        cache = InMemoryCache()

        result = cache.get("What is Spark?")

        self.assertIsNone(result)

        stats = cache.stats()

        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 0)

    def test_cache_hit(self):
        cache = InMemoryCache()

        cache.set(
            "What is Spark?",
            {"answer": "Spark is a distributed computing engine."},
        )

        result = cache.get("What is Spark?")

        self.assertIsNotNone(result)
        self.assertEqual(
            result["answer"],
            "Spark is a distributed computing engine.",
        )

        stats = cache.stats()

        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)

    def test_question_normalization(self):
        cache = InMemoryCache()

        cache.set(
            "  What IS Spark?  ",
            {"answer": "cached"},
        )

        result = cache.get("what is spark?")

        self.assertEqual(
            result["answer"],
            "cached",
        )

    def test_cache_clear(self):
        cache = InMemoryCache()

        cache.set(
            "What is Spark?",
            {"answer": "cached"},
        )

        cache.clear()

        self.assertIsNone(
            cache.get("What is Spark?")
        )


if __name__ == "__main__":
    unittest.main()
