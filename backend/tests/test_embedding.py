import sys
from pathlib import Path

# Add backend/src to path if run standalone
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repository.products import Product, ProductSpec
from rag.embedding import Embedding, ProductChunk


class TestProductChunk(unittest.TestCase):
    def test_chunk_to_dict_and_to_langchain_document(self):
        chunk = ProductChunk(
            chunk_id="p1_summary",
            product_id="p1",
            chunk_type="summary",
            text="نام کالا: تست محصول",
            metadata={"brand_name": "سامسونگ", "category_name": "موبایل"},
        )
        data = chunk.to_dict()
        self.assertEqual(data["chunk_id"], "p1_summary")
        self.assertEqual(data["product_id"], "p1")
        self.assertEqual(data["chunk_type"], "summary")
        self.assertEqual(data["text"], "نام کالا: تست محصول")
        self.assertEqual(data["metadata"]["brand_name"], "سامسونگ")

        doc = chunk.to_langchain_document()
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.page_content, chunk.text)
        self.assertEqual(doc.id, "p1_summary")
        self.assertEqual(doc.metadata["chunk_id"], "p1_summary")
        self.assertEqual(doc.metadata["product_id"], "p1")
        self.assertEqual(doc.metadata["chunk_type"], "summary")
        self.assertEqual(doc.metadata["brand_name"], "سامسونگ")
        self.assertEqual(doc.metadata["category_name"], "موبایل")


class TestChunkingStrategy(unittest.TestCase):
    def setUp(self):
        self.sample_specs = [
            ProductSpec(key="حافظه داخلی", value="256 گیگابایت", section="حافظه"),
            ProductSpec(key="رم", value="8 گیگابایت", section="حافظه"),
            ProductSpec(key="دوربین اصلی", value="50 مگاپیکسل", section="دوربین"),
            ProductSpec(key="ظرفیت باتری", value="5000 میلی‌آمپر", section="باتری"),
        ]
        self.product = Product(
            product_id="prod_123",
            persian_title="گوشی سامسونگ گلکسی S24",
            category_name="گوشی موبایل",
            brand_name="سامسونگ",
            model="S24",
            specs=self.sample_specs,
        )

    def test_chunk_from_product_object(self):
        chunks = Embedding.chunk(self.product, max_specs_per_chunk=2)
        self.assertTrue(len(chunks) >= 2)

        # First chunk must be summary
        summary = chunks[0]
        self.assertEqual(summary.chunk_type, "summary")
        self.assertEqual(summary.chunk_id, "prod_123_summary")
        self.assertEqual(summary.product_id, "prod_123")
        self.assertIn("نام کالا: گوشی سامسونگ گلکسی S24", summary.text)
        self.assertIn("برند: سامسونگ", summary.text)
        self.assertIn("دسته‌بندی: گوشی موبایل", summary.text)
        self.assertIn("مدل: S24", summary.text)
        self.assertIn("مشخصات کلیدی:", summary.text)
        self.assertIn("- حافظه داخلی: 256 گیگابایت", summary.text)

        # Subsequent chunks are specs chunks
        specs_chunks = chunks[1:]
        for sc in specs_chunks:
            self.assertEqual(sc.chunk_type, "specs")
            self.assertEqual(sc.product_id, "prod_123")
            self.assertIn("محصول: گوشی سامسونگ گلکسی S24", sc.text)
            self.assertIn("دسته‌بندی: گوشی موبایل", sc.text)
            self.assertIn("برند: سامسونگ", sc.text)
            self.assertIn("بخش:", sc.text)

    def test_chunk_from_dict_input(self):
        dict_product = {
            "product_id": "prod_dict",
            "persian_title": "لپ‌تاپ لنوو ThinkPad",
            "brand_name": "لنوو",
            "category_name": "لپ‌تاپ",
            "model": "T14",
            "specs": [
                {"key": "پردازنده", "value": "Core i7", "section": "پردازنده"},
                {"key": "رم", "value": "16 گیگابایت", "section": "حافظه"},
            ],
        }
        chunks = Embedding.chunk(dict_product)
        self.assertEqual(len(chunks), 3)  # summary + 2 section chunks
        self.assertEqual(chunks[0].chunk_type, "summary")
        self.assertEqual(chunks[0].product_id, "prod_dict")
        self.assertIn("لپ‌تاپ لنوو ThinkPad", chunks[0].text)

    def test_chunk_from_dict_with_json_string_specs(self):
        dict_product = {
            "product_id": "prod_json_str",
            "persian_title": "هدفون بی سیم سونی",
            "brand_name": "سونی",
            "category_name": "هدفون",
            "specs": '[{"key": "نوع اتصال", "value": "بلوتوث", "section": "اتصالات"}]',
        }
        chunks = Embedding.chunk(dict_product)
        self.assertEqual(len(chunks), 2)  # 1 summary, 1 specs chunk
        self.assertEqual(chunks[1].chunk_type, "specs")
        self.assertIn("نوع اتصال: بلوتوث", chunks[1].text)

    def test_chunk_product_without_specs(self):
        minimal_product = Product(
            product_id="prod_no_specs",
            persian_title="کتاب آموزش پایتون",
            category_name="کتاب",
        )
        chunks = Embedding.chunk(minimal_product)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "summary")
        self.assertEqual(chunks[0].chunk_id, "prod_no_specs_summary")
        self.assertIn("نام کالا: کتاب آموزش پایتون", chunks[0].text)

    def test_chunk_to_documents(self):
        docs = Embedding.chunk_to_documents(self.product)
        self.assertTrue(len(docs) > 0)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].id, "prod_123_summary")
        self.assertEqual(docs[0].metadata["product_id"], "prod_123")
        self.assertEqual(docs[0].metadata["chunk_type"], "summary")


class TestEmbeddingClient(unittest.TestCase):
    def test_implements_langchain_interface(self):
        embedder = Embedding(api_key="")
        self.assertIsInstance(embedder, Embeddings)
        self.assertTrue(callable(getattr(embedder, "embed_documents", None)))
        self.assertTrue(callable(getattr(embedder, "embed_query", None)))

    def test_fallback_zero_vector_when_no_api_key(self):
        embedder_small = Embedding(model_name="text-embedding-3-small", api_key="")
        vectors_small = embedder_small.embed_documents(["متن اول", "متن دوم"])
        self.assertEqual(len(vectors_small), 2)
        self.assertEqual(len(vectors_small[0]), 1536)
        self.assertEqual(vectors_small[0], [0.0] * 1536)

        query_vec = embedder_small.embed_query("جستجوی تست")
        self.assertEqual(len(query_vec), 1536)
        self.assertEqual(query_vec, [0.0] * 1536)

        # Large model dimension fallback
        embedder_large = Embedding(model_name="text-embedding-3-large", api_key="")
        vectors_large = embedder_large.embed_documents(["تست"])
        self.assertEqual(len(vectors_large[0]), 3072)

    def test_embed_empty_list(self):
        embedder = Embedding(api_key="")
        self.assertEqual(embedder.embed_documents([]), [])

    def test_embed_convenience_method(self):
        embedder = Embedding(api_key="")
        # Single string -> returns single vector
        single = embedder.embed("متن آزمایشی")
        self.assertIsInstance(single, list)
        self.assertEqual(len(single), 1536)

        # List of strings -> returns list of vectors
        multi = embedder.embed(["متن 1", "متن 2"])
        self.assertIsInstance(multi, list)
        self.assertEqual(len(multi), 2)
        self.assertEqual(len(multi[0]), 1536)

    @patch("rag.embedding.httpx.Client")
    def test_mocked_http_api_call(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        mock_client.post.return_value = mock_resp

        embedder = Embedding(api_key="secret-boxai-key", base_url="https://ai.boxapi.ir/ai/v1")
        results = embedder.embed_documents(["متن اول", "متن دوم"])

        self.assertEqual(results, [[0.1, 0.2], [0.3, 0.4]])
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://ai.boxapi.ir/ai/v1/embeddings")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-boxai-key")
        self.assertEqual(kwargs["json"]["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["json"]["input"], ["متن اول", "متن دوم"])

    def test_create_vector_store_and_search(self):
        # Using mock vector with non-zero values to test InMemoryVectorStore indexing
        embedder = Embedding(api_key="")
        product = Product(
            product_id="p100",
            persian_title="هدفون بلوتوثی هایلو",
            category_name="هدفون",
            brand_name="هایلو",
        )
        docs = Embedding.chunk_to_documents(product)

        # Mock embed_documents and embed_query to return unit vectors
        mock_vec = [1.0] + [0.0] * 1535
        with patch.object(embedder, "embed_documents", return_value=[mock_vec for _ in docs]), \
             patch.object(embedder, "embed_query", return_value=mock_vec):
            vector_store = embedder.create_vector_store(docs)
            self.assertIsNotNone(vector_store)

            # Check that similarity search functions when numpy is available
            try:
                retrieved = vector_store.similarity_search("هدفون هایلو", k=1)
                self.assertEqual(len(retrieved), 1)
                self.assertEqual(retrieved[0].metadata["product_id"], "p100")
            except ImportError:
                # If numpy is not yet installed in the active environment, vector store addition succeeded
                self.assertTrue(len(vector_store.store) > 0)

    def test_qdrant_product_store_disk(self):
        import tempfile
        import shutil
        from rag.qdrant_store import QdrantProductStore

        temp_dir = tempfile.mkdtemp()
        try:
            embedder = Embedding(api_key="")
            mock_vec = [0.1] * 1536
            with patch.object(embedder, "embed_documents", return_value=[mock_vec]), \
                 patch.object(embedder, "embed_query", return_value=mock_vec):

                store = QdrantProductStore(
                    embedding=embedder,
                    storage_path=temp_dir,
                    collection_name="test_products",
                )

                chunk = ProductChunk(
                    chunk_id="chunk_test_1",
                    product_id="prod_1",
                    chunk_type="summary",
                    text="گوشی تست",
                    metadata={"brand_name": "اپل"},
                )
                added = store.add_chunks([chunk])
                self.assertEqual(added, 1)
                self.assertEqual(store.count(), 1)

                results = store.search("گوشی", limit=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].metadata["product_id"], "prod_1")
                self.assertEqual(results[0].metadata["brand_name"], "اپل")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
