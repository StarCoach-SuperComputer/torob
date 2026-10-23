import re
from typing import List, Dict, Any, Optional, Set
from langchain_core.documents import Document
from .qdrant_store import QdrantProductStore
from .reranker import Reranker


def rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    limit: int = 10,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to merge and rank results from dense
    semantic search and sparse keyword search.

    RRF score = sum(weight / (k + rank))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Score dense documents
    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = doc.id or doc.metadata.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + (dense_weight / (k + rank))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Score sparse / keyword documents
    for rank, doc in enumerate(sparse_docs, start=1):
        doc_id = doc.id or doc.metadata.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort doc_ids by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results: List[Document] = []
    for doc_id in sorted_ids[:limit]:
        doc = doc_map[doc_id]
        # Attach combined rrf_score to metadata
        merged_meta = dict(doc.metadata)
        merged_meta["rrf_score"] = scores[doc_id]
        results.append(
            Document(
                page_content=doc.page_content,
                metadata=merged_meta,
                id=doc_id,
            )
        )

    return results


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (Qdrant vector embeddings)
    with sparse full-text / keyword search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        store: QdrantProductStore,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self.store = store
        self.reranker = reranker or Reranker()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        candidate_multiplier: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
        chunk_type: Optional[str] = None,
        group_by_product: bool = False,
        use_reranker: bool = True,
    ) -> List[Document]:
        """
        Executes hybrid search:
        1. Dense semantic search via embedding vector similarity
        2. Sparse keyword matching via Qdrant full-text search
        3. Fuses candidates via Reciprocal Rank Fusion (RRF)
        4. Optionally groups/deduplicates multiple chunks of the same product
        5. Precision Reranking via Cross-Encoder / Reranker model
        """
        candidate_k = max(limit * candidate_multiplier, 15)

        # 1. Fetch dense candidates
        dense_results = self.store.search(
            query=query,
            limit=candidate_k,
            filter_dict=filter_dict,
            chunk_type=chunk_type,
        )

        # 2. Fetch keyword / full-text candidates
        sparse_results = self.store.text_search(
            query=query,
            limit=candidate_k,
            filter_dict=filter_dict,
            chunk_type=chunk_type,
        )

        # 3. Merge candidates using Reciprocal Rank Fusion (RRF)
        fused = rrf_merge(
            dense_docs=dense_results,
            sparse_docs=sparse_results,
            k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            limit=candidate_k if group_by_product else limit,
        )

        # 4. Optional product-level deduplication
        candidates = fused
        if group_by_product:
            seen_products: Set[str] = set()
            deduped: List[Document] = []
            for doc in fused:
                pid = doc.metadata.get("product_id")
                if pid:
                    if pid in seen_products:
                        continue
                    seen_products.add(pid)
                deduped.append(doc)
            candidates = deduped

        # 5. Precision Reranking pass
        if use_reranker and self.reranker and candidates:
            # Rerank the top candidates using cross-encoder scoring
            rerank_pool = candidates[: candidate_k]
            reranked = self.reranker.rerank(query=query, documents=rerank_pool, top_n=limit)
            return reranked

        return candidates[:limit]

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """LangChain standard retriever interface compatibility."""
        return self.retrieve(query=query, **kwargs)

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """Runnable interface compatibility."""
        return self.retrieve(query=query, **kwargs)
