import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag.reranker import Reranker


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = Reranker()

    def test_local_rerank_prioritizes_exact_matches(self):
        query = "کابل افشان مشهد روکش pvc"

        doc1 = Document(
            page_content="اسپیکر بلوتوثی با روکش چرمی متری",
            metadata={"product_id": "p1", "persian_title": "اسپیکر قابل حمل"},
            id="p1",
        )
        doc2 = Document(
            page_content="کابل افشان مشهد با روکش pvc کیفیت عالی",
            metadata={"product_id": "p2", "persian_title": "کابل افشان مشهد با روکش PVC"},
            id="p2",
        )
        doc3 = Document(
            page_content="سیم و کابل برق ساختمانی متفرقه",
            metadata={"product_id": "p3", "persian_title": "سیم تک رشته"},
            id="p3",
        )

        candidates = [doc1, doc2, doc3]
        reranked = self.reranker.rerank(query=query, documents=candidates, top_n=3)

        self.assertEqual(len(reranked), 3)
        # doc2 should be ranked 1st with the highest rerank score
        self.assertEqual(reranked[0].id, "p2")
        self.assertGreater(reranked[0].metadata["rerank_score"], reranked[1].metadata["rerank_score"])
        self.assertGreater(reranked[0].metadata["rerank_score"], reranked[2].metadata["rerank_score"])

    def test_rerank_empty_documents(self):
        res = self.reranker.rerank(query="تست", documents=[])
        self.assertEqual(res, [])

    @patch("httpx.Client.post")
    def test_remote_rerank_api_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.12},
            ]
        }
        mock_post.return_value = mock_response

        reranker = Reranker(api_key="test-key", base_url="https://fake-rerank.api")
        docs = [
            Document(page_content="محتوا ۱", metadata={"product_id": "p1"}, id="p1"),
            Document(page_content="محتوا ۲", metadata={"product_id": "p2"}, id="p2"),
        ]

        reranked = reranker.rerank(query="جستجو", documents=docs, top_n=2)
        self.assertEqual(reranked[0].id, "p2")
        self.assertEqual(reranked[0].metadata["rerank_score"], 0.98)
        self.assertEqual(reranked[1].id, "p1")
        self.assertEqual(reranked[1].metadata["rerank_score"], 0.12)


if __name__ == "__main__":
    unittest.main()
