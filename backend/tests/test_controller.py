import sys
from pathlib import Path

# Add backend/src to path if run standalone
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from controller.home_controller import get_retriever, SearchRequest


class TestHomeController(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("controller.home_controller.get_retriever")
    def test_search_post_endpoint(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever

        # Sample retrieved document
        mock_doc = Document(
            page_content="نام کالا: گوشی گلکسی S24 سامسونگ\nبرند: سامسونگ",
            metadata={
                "product_id": "p100",
                "chunk_id": "p100_summary",
                "chunk_type": "summary",
                "persian_title": "گوشی گلکسی S24 سامسونگ",
                "brand_name": "سامسونگ",
                "category_name": "موبایل",
                "model": "S24",
                "rrf_score": 0.032,
            },
            id="p100_summary",
        )
        mock_retriever.retrieve.return_value = [mock_doc]

        payload = {
            "query": "سامسونگ اس ۲۴",
            "limit": 5,
            "group_by_product": True,
        }
        response = self.client.post("/search", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query"], "سامسونگ اس ۲۴")
        self.assertEqual(data["total_results"], 1)
        self.assertEqual(data["results"][0]["product_id"], "p100")
        self.assertEqual(data["results"][0]["brand_name"], "سامسونگ")
        self.assertAlmostEqual(data["results"][0]["raw_rrf_score"], 0.032, places=3)
        self.assertGreater(data["results"][0]["score"], 0.9)

    @patch("controller.home_controller.get_retriever")
    def test_search_get_endpoint(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever

        mock_doc = Document(
            page_content="لپ تاپ ایسوس",
            metadata={"product_id": "p200", "persian_title": "لپ تاپ ایسوس", "rrf_score": 0.025},
            id="p200",
        )
        mock_retriever.retrieve.return_value = [mock_doc]

        response = self.client.get("/search?q=ایسوس&limit=2")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query"], "ایسوس")
        self.assertEqual(data["total_results"], 1)
        self.assertEqual(data["results"][0]["product_id"], "p200")


if __name__ == "__main__":
    unittest.main()
