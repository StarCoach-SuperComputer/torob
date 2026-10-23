import os
import uuid
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from langchain_core.documents import Document
from .embedding import Embedding, ProductChunk
from repository.products import Product


class QdrantProductStore:
    """
    On-disk Qdrant vector database store for product chunks.
    Uses local file-based storage path (embedded SQLite/storage engines in qdrant-client).
    """

    DEFAULT_COLLECTION_NAME = "products"
    DEFAULT_STORAGE_PATH = "qdrant_storage"

    def __init__(
        self,
        embedding: Optional[Embedding] = None,
        url: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: Optional[int] = None,
        distance: rest_models.Distance = rest_models.Distance.COSINE,
    ):
        self.embedding = embedding or Embedding()
        self.collection_name = collection_name
        self.url = url or os.getenv("QDRANT_URL")
        self.storage_path = str(storage_path or os.getenv("QDRANT_STORAGE_PATH") or self.DEFAULT_STORAGE_PATH)

        # Resolve vector dimension: 1536 for text-embedding-3-small, 3072 for large, or custom
        if vector_size:
            self.vector_size = vector_size
        else:
            self.vector_size = 3072 if "large" in self.embedding.model_name else 1536

        self.distance = distance

        # If url is provided, connect to server; otherwise use local on-disk storage path
        if self.url:
            # Bypass socks proxy for local connections (avoids httpx socks proxy parse error)
            import httpx
            from qdrant_client.http.api_client import AsyncApis, SyncApis

            client_kwargs = {}
            if "localhost" in self.url or "127.0.0.1" in self.url:
                # Disable environment proxy for local Qdrant instance
                client_kwargs["trust_env"] = False

            self.client = QdrantClient(url=self.url, **client_kwargs)
        else:
            self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the collection and full-text index if it does not already exist."""
        collections_response = self.client.get_collections()
        existing_names = [col.name for col in collections_response.collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

        # Ensure full-text index on page_content and persian_title for sparse / keyword search
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_content",
                field_schema=rest_models.TextIndexParams(
                    type="text",
                    tokenizer=rest_models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                ),
            )
        except Exception:
            # Index might already exist or be created
            pass

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """
        Qdrant point IDs must be unsigned integers or valid UUIDs.
        Generates a deterministic UUID5 from the chunk_id string.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def add_chunks(self, chunks: List[ProductChunk], batch_size: int = 64) -> int:
        """
        Embeds and writes a list of ProductChunk objects to local Qdrant on disk.
        """
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = self.embedding.embed_documents(texts)

            points = []
            for chunk, vector in zip(batch, vectors):
                point_id = self._chunk_id_to_uuid(chunk.chunk_id)
                payload = dict(chunk.metadata)
                payload.update({
                    "chunk_id": chunk.chunk_id,
                    "product_id": chunk.product_id,
                    "chunk_type": chunk.chunk_type,
                    "page_content": chunk.text,
                })

                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_added += len(points)

        return total_added

    def add_documents(self, documents: List[Document], batch_size: int = 64) -> int:
        """
        Embeds and stores LangChain Document objects in local Qdrant on disk.
        """
        if not documents:
            return 0

        chunks = []
        for doc in documents:
            chunk_id = doc.id or doc.metadata.get("chunk_id") or str(uuid.uuid4())
            product_id = doc.metadata.get("product_id", "")
            chunk_type = doc.metadata.get("chunk_type", "summary")
            chunks.append(
                ProductChunk(
                    chunk_id=chunk_id,
                    product_id=product_id,
                    chunk_type=chunk_type,
                    text=doc.page_content,
                    metadata=doc.metadata,
                )
            )
        return self.add_chunks(chunks, batch_size=batch_size)

    def add_product(self, product: Union[Product, Dict[str, Any]], max_specs_per_chunk: int = 12) -> int:
        """
        Chunks a product and adds all generated chunks to the local Qdrant database.
        """
        chunks = Embedding.chunk(product, max_specs_per_chunk=max_specs_per_chunk)
        return self.add_chunks(chunks)

    def add_products(
        self,
        products: List[Union[Product, Dict[str, Any]]],
        max_specs_per_chunk: int = 12,
        batch_size: int = 64,
    ) -> int:
        """
        Chunks multiple products and adds them in batches to Qdrant.
        """
        all_chunks: List[ProductChunk] = []
        for p in products:
            chunks = Embedding.chunk(p, max_specs_per_chunk=max_specs_per_chunk)
            all_chunks.extend(chunks)

        return self.add_chunks(all_chunks, batch_size=batch_size)

    def search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Performs semantic similarity search against the local on-disk Qdrant collection.
        Returns matching results as LangChain Document instances.
        """
        query_vector = self.embedding.embed_query(query)

        qdrant_filter = None
        conditions = []

        if chunk_type:
            conditions.append(
                rest_models.FieldCondition(
                    key="chunk_type",
                    match=rest_models.MatchValue(value=chunk_type),
                )
            )

        if filter_dict:
            for k, v in filter_dict.items():
                conditions.append(
                    rest_models.FieldCondition(
                        key=k,
                        match=rest_models.MatchValue(value=v),
                    )
                )

        if conditions:
            qdrant_filter = rest_models.Filter(must=conditions)

        # qdrant_client query_points or search
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
            )

        results: List[Document] = []
        for pt in points:
            payload = pt.payload or {}
            content = payload.get("page_content", "")
            meta = {k: v for k, v in payload.items() if k != "page_content"}
            meta["score"] = pt.score
            results.append(
                Document(
                    page_content=content,
                    metadata=meta,
                    id=str(pt.id),
                )
            )

        return results

    def text_search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Performs full-text keyword search directly on Qdrant payload fields.
        """
        conditions = []

        # Full-text match condition on page_content
        # In Persian, filter out common short stopwords that cause noisy matches
        stopwords = {"با", "از", "در", "به", "تا", "که", "بر", "برای", "و", "یا"}
        clean_q = query.strip()
        if clean_q:
            words = [w for w in clean_q.split() if len(w) > 1 and w not in stopwords]
            if len(words) > 1:
                # Match either the full phrase or any of the meaningful keywords
                word_conditions = [
                    rest_models.FieldCondition(
                        key="page_content",
                        match=rest_models.MatchText(text=w),
                    )
                    for w in words
                ]
                conditions.append(rest_models.Filter(should=word_conditions))
            elif words:
                conditions.append(
                    rest_models.FieldCondition(
                        key="page_content",
                        match=rest_models.MatchText(text=words[0]),
                    )
                )

        if chunk_type:
            conditions.append(
                rest_models.FieldCondition(
                    key="chunk_type",
                    match=rest_models.MatchValue(value=chunk_type),
                )
            )

        if filter_dict:
            for k, v in filter_dict.items():
                conditions.append(
                    rest_models.FieldCondition(
                        key=k,
                        match=rest_models.MatchValue(value=v),
                    )
                )

        qdrant_filter = rest_models.Filter(must=conditions) if conditions else None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: List[Document] = []
        for rec in records:
            payload = rec.payload or {}
            content = payload.get("page_content", "")
            meta = {k: v for k, v in payload.items() if k != "page_content"}
            meta["score"] = 1.0  # standard sparse hit score
            results.append(
                Document(
                    page_content=content,
                    metadata=meta,
                    id=str(rec.id),
                )
            )

        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        candidate_multiplier: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
        chunk_type: Optional[str] = None,
        group_by_product: bool = False,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[Document]:
        """
        Convenience method to execute hybrid search directly on the store.
        """
        from .retriever import HybridRetriever

        retriever = HybridRetriever(
            store=self,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        return retriever.retrieve(
            query=query,
            limit=limit,
            candidate_multiplier=candidate_multiplier,
            filter_dict=filter_dict,
            chunk_type=chunk_type,
            group_by_product=group_by_product,
        )

    def count(self) -> int:
        """Returns total count of vectors stored in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
