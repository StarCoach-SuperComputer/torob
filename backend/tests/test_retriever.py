import sys
from pathlib import Path

# Add backend/src to path if run standalone
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from rag.retriever import HybridRetriever, rrf_merge
from rag.embedding import Embedding, ProductChunk
from rag.qdrant_store import QdrantProductStore


class TestRRFMerge(unittest.TestCase):
    def test_rrf_scoring_and_ranking(self):
        doc1 = Document(page_content="گوشی سامسونگ S24", metadata={"product_id": "p1"}, id="p1")
        doc2 = Document(page_content="گوشی شیائومی 14", metadata={"product_id": "p2"}, id="p2")
        doc3 = Document(page_content="آیفون 15", metadata={"product_id": "p3"}, id="p3")

        # Dense ranking: p1 (rank 1), p2 (rank 2)
        dense_docs = [doc1, doc2]
        # Sparse ranking: p2 (rank 1), p3 (rank 2)
        sparse_docs = [doc2, doc3]

        fused = rrf_merge(
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            k=60,
            limit=3,
        )

        self.assertEqual(len(fused), 3)
        # p2 is present in both lists, so it must rank highest
        self.assertEqual(fused[0].id, "p2")
        self.assertIn("rrf_score", fused[0].metadata)
        self.assertGreater(fused[0].metadata["rrf_score"], fused[1].metadata["rrf_score"])


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=QdrantProductStore)
        self.retriever = HybridRetriever(
            store=self.mock_store,
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
        )

    def test_retrieve_fuses_dense_and_sparse(self):
        doc_dense = Document(page_content="سامسونگ اس ۲۴", metadata={"product_id": "p1"}, id="p1")
        doc_sparse = Document(page_content="سامسونگ گلکسی S24", metadata={"product_id": "p1"}, id="p1")
        doc_other = Document(page_content="شیائومی", metadata={"product_id": "p2"}, id="p2")

        self.mock_store.search.return_value = [doc_dense, doc_other]
        self.mock_store.text_search.return_value = [doc_sparse]

        results = self.retriever.retrieve(query="سامسونگ", limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "p1")
        self.assertIn("rrf_score", results[0].metadata)

        # Verify Qdrant methods were called with query
        self.mock_store.search.assert_called_once()
        self.mock_store.text_search.assert_called_once()

    def test_retrieve_group_by_product(self):
        # Multiple chunks belonging to the same product_id
        chunk1 = Document(page_content="خلاصه محصول", metadata={"product_id": "prod_A", "chunk_id": "c1"}, id="c1")
        chunk2 = Document(page_content="مشخصات محصول", metadata={"product_id": "prod_A", "chunk_id": "c2"}, id="c2")
        chunk3 = Document(page_content="محصول دیگر", metadata={"product_id": "prod_B", "chunk_id": "c3"}, id="c3")

        self.mock_store.search.return_value = [chunk1, chunk2, chunk3]
        self.mock_store.text_search.return_value = []

        # With group_by_product=True, prod_A should only appear once
        deduped = self.retriever.retrieve(query="تست", limit=5, group_by_product=True)
        self.assertEqual(len(deduped), 2)
        pids = [d.metadata["product_id"] for d in deduped]
        self.assertEqual(pids, ["prod_A", "prod_B"])


if __name__ == "__main__":
    unittest.main()
