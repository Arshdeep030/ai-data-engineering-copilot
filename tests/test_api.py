"""Unit and contract tests for FastAPI application."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app


class TestCopilotAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_docs_endpoints(self):
        swagger_response = self.client.get("/docs")
        self.assertEqual(swagger_response.status_code, 200)

        redoc_response = self.client.get("/redoc")
        self.assertEqual(redoc_response.status_code, 200)

    def test_query_validation_empty_string(self):
        response = self.client.post("/query", json={"question": ""})
        self.assertEqual(response.status_code, 422)

    def test_query_validation_missing_field(self):
        response = self.client.post("/query", json={})
        self.assertEqual(response.status_code, 422)

    @patch("app.query_rag")
    def test_query_success_contract(self, mock_query_rag):
        mock_query_rag.return_value = {
            "question": "What is Spark?",
            "answer": "Apache Spark is a distributed engine.",
            "sources": [
                {
                    "id": "spark_apache_spark_000",
                    "section": "Apache Spark",
                    "source": "spark.md",
                }
            ],
            "latency_seconds": 1.25,
        }

        response = self.client.post(
            "/query", json={"question": "What is Spark?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], "What is Spark?")
        self.assertEqual(data["answer"], "Apache Spark is a distributed engine.")
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["id"], "spark_apache_spark_000")
        self.assertEqual(data["latency_seconds"], 1.25)
        self.assertFalse(data["cache_hit"])
        self.assertTrue(len(data["request_id"]) > 0)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("runtime", data)
        self.assertIn("cache", data)
        self.assertIn("requests_total", data["runtime"])
        self.assertIn("hits", data["cache"])
        self.assertIn("hit_rate", data["cache"])


if __name__ == "__main__":
    unittest.main()
