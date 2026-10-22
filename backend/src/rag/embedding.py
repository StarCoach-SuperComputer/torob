import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
import httpx
from repository.products import Product, ProductSpec


@dataclass
class ProductChunk:
    chunk_id: str
    product_id: str
    chunk_type: str  # "summary" or "specs"
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_langchain_document(self) -> Document:
        """Converts ProductChunk into a LangChain Document."""
        doc_metadata = dict(self.metadata)
        doc_metadata.update({
            "chunk_id": self.chunk_id,
            "product_id": self.product_id,
            "chunk_type": self.chunk_type,
        })
        return Document(
            page_content=self.text,
            metadata=doc_metadata,
            id=self.chunk_id,
        )


class Embedding(Embeddings):
    """
    LangChain-compliant Embeddings client using BoxAI / OpenAI compatible endpoints.
    Implements `embed_documents` and `embed_query` from `langchain_core.embeddings.Embeddings`.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("BOXAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("BOXAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://ai.boxapi.ir/ai/v1"
        ).rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain interface for embedding a list of documents."""
        if not texts:
            return []

        if not self.api_key:
            dim = 1536 if "large" not in self.model_name else 3072
            return [[0.0] * dim for _ in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def embed_query(self, text: str) -> List[float]:
        """LangChain interface for embedding a single user query."""
        results = self.embed_documents([text])
        return results[0] if results else []

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Convenience method accepting either a single string or a list of strings."""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def create_vector_store(self, documents: Optional[List[Document]] = None) -> InMemoryVectorStore:
        """Creates an in-memory LangChain vector store using this embedding client."""
        store = InMemoryVectorStore(embedding=self)
        if documents:
            store.add_documents(documents)
        return store

    def create_qdrant_store(
        self,
        storage_path: Optional[Union[str, Any]] = None,
        collection_name: str = "products",
        documents: Optional[List[Document]] = None,
    ) -> Any:
        """
        Creates an on-disk QdrantProductStore using this embedding instance.
        """
        from .qdrant_store import QdrantProductStore

        store = QdrantProductStore(
            embedding=self,
            storage_path=storage_path,
            collection_name=collection_name,
        )
        if documents:
            store.add_documents(documents)
        return store

    @classmethod
    def chunk(
        cls,
        product: Union[Product, Dict[str, Any]],
        max_specs_per_chunk: int = 12,
    ) -> List[ProductChunk]:
        """
        Chunking strategy tailored for e-commerce product search and technical RAG retrieval.
        Generates both high-level semantic summary chunks and granular technical spec chunks.
        """
        if isinstance(product, dict):
            product_id = product.get("product_id", "")
            persian_title = product.get("persian_title", "")
            brand_name = product.get("brand_name")
            category_name = product.get("category_name")
            model = product.get("model")
            raw_specs = product.get("specs", [])
            specs_list: List[ProductSpec] = []
            if isinstance(raw_specs, str):
                try:
                    raw_specs = json.loads(raw_specs)
                except Exception:
                    raw_specs = []
            if isinstance(raw_specs, list):
                for s in raw_specs:
                    if isinstance(s, dict):
                        specs_list.append(
                            ProductSpec(
                                key=s.get("key", ""),
                                value=s.get("value", ""),
                                section=s.get("section"),
                            )
                        )
        else:
            product_id = product.product_id
            persian_title = product.persian_title
            brand_name = product.brand_name
            category_name = product.category_name
            model = product.model
            specs_list = product.specs or []

        chunks: List[ProductChunk] = []

        # 1. Primary Summary Chunk
        summary_lines = []
        if persian_title:
            summary_lines.append(f"نام کالا: {persian_title}")
        if category_name:
            summary_lines.append(f"دسته‌بندی: {category_name}")
        if brand_name:
            summary_lines.append(f"برند: {brand_name}")
        if model:
            summary_lines.append(f"مدل: {model}")

        # Add top specs to summary chunk
        if specs_list:
            summary_lines.append("مشخصات کلیدی:")
            for spec in specs_list[:8]:
                summary_lines.append(f"- {spec.key}: {spec.value}")

        summary_text = "\n".join(summary_lines)
        chunks.append(
            ProductChunk(
                chunk_id=f"{product_id}_summary",
                product_id=product_id,
                chunk_type="summary",
                text=summary_text,
                metadata={
                    "product_id": product_id,
                    "persian_title": persian_title,
                    "brand_name": brand_name,
                    "category_name": category_name,
                    "model": model,
                    "chunk_type": "summary",
                },
            )
        )

        # 2. Detailed Specs Chunks (grouped by section or bounded batches)
        if specs_list:
            section_map: Dict[str, List[ProductSpec]] = {}
            for s in specs_list:
                sec = s.section or "مشخصات"
                section_map.setdefault(sec, []).append(s)

            chunk_idx = 1
            for section_name, specs in section_map.items():
                for i in range(0, len(specs), max_specs_per_chunk):
                    batch = specs[i : i + max_specs_per_chunk]
                    spec_lines = [
                        f"محصول: {persian_title}",
                    ]
                    if category_name:
                        spec_lines.append(f"دسته‌بندی: {category_name}")
                    if brand_name:
                        spec_lines.append(f"برند: {brand_name}")

                    spec_lines.append(f"بخش: {section_name}")
                    for s in batch:
                        spec_lines.append(f"- {s.key}: {s.value}")

                    spec_chunk_text = "\n".join(spec_lines)
                    chunks.append(
                        ProductChunk(
                            chunk_id=f"{product_id}_specs_{chunk_idx}",
                            product_id=product_id,
                            chunk_type="specs",
                            text=spec_chunk_text,
                            metadata={
                                "product_id": product_id,
                                "persian_title": persian_title,
                                "brand_name": brand_name,
                                "category_name": category_name,
                                "section": section_name,
                                "chunk_type": "specs",
                            },
                        )
                    )
                    chunk_idx += 1

        return chunks

    @classmethod
    def chunk_to_documents(
        cls,
        product: Union[Product, Dict[str, Any]],
        max_specs_per_chunk: int = 12,
    ) -> List[Document]:
        """Helper to directly produce LangChain Document objects from a product."""
        chunks = cls.chunk(product, max_specs_per_chunk=max_specs_per_chunk)
        return [c.to_langchain_document() for c in chunks]
