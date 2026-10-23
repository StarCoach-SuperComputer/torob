import os
from typing import List, Dict, Any, Optional
import httpx
from langchain_core.documents import Document


class Reranker:
    """
    Cross-Encoder / Reranker client for precision scoring of candidate documents.
    Supports BoxAI / Cohere / Jina / BGE compatible rerank APIs,
    with an intelligent lexical/semantic fallback when offline or without an API key.
    """

    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("BOXAI_API_KEY") or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("BOXAI_BASE_URL")
            or os.getenv("RERANK_BASE_URL")
            or "https://ai.boxapi.ir/ai/v1"
        ).rstrip("/")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        Reranks a list of candidate documents against the query.
        Returns the documents sorted by reranker relevance score descending.
        """
        if not documents:
            return []

        top_n = top_n or len(documents)

        # If API key is present, use external cross-encoder rerank API (Cohere / BoxAI / Jina standard format)
        if self.api_key:
            try:
                reranked_docs = self._call_remote_rerank(query, documents, top_n)
                if reranked_docs:
                    return reranked_docs
            except Exception:
                # Fall back gracefully to local precision re-scoring
                pass

        # Local intelligent lexical + overlap precision re-scoring fallback
        return self._local_rerank(query, documents, top_n)

    def _call_remote_rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Document]:
        """
        Calls /rerank endpoint (Cohere / BGE / BoxAI standard interface).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        doc_texts = [doc.page_content for doc in documents]
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": top_n,
        }

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{self.base_url}/rerank", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            reranked: List[Document] = []
            for item in sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True):
                idx = item["index"]
                orig_doc = documents[idx]
                meta = dict(orig_doc.metadata)
                meta["rerank_score"] = float(item.get("relevance_score", 0.0))
                reranked.append(
                    Document(
                        page_content=orig_doc.page_content,
                        metadata=meta,
                        id=orig_doc.id,
                    )
                )
            return reranked

    def _local_rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Document]:
        """
        Precision lexical & token-overlap re-scorer used when offline or without API key.
        Evaluates exact phrase matching, token coverage, title weighting, and specs alignment.
        """
        stopwords = {"با", "از", "در", "به", "تا", "که", "بر", "برای", "و", "یا", "یک", "این"}
        query_clean = query.strip()
        query_lower = query_clean.lower()
        query_tokens = [t for t in query_lower.split() if t not in stopwords and len(t) > 1]

        scored_docs = []
        for doc in documents:
            content = doc.page_content.lower()
            title = str(doc.metadata.get("persian_title", "")).lower()
            category = str(doc.metadata.get("category_name", "")).lower()
            brand = str(doc.metadata.get("brand_name", "")).lower()
            initial_score = float(doc.metadata.get("rrf_score", doc.metadata.get("score", 0.0)))

            score = 0.0

            # 1. Exact phrase match bonus
            if query_lower in title:
                score += 0.50
            elif query_lower in content:
                score += 0.30

            # 2. Token overlap & title coverage
            if query_tokens:
                title_matches = sum(1 for t in query_tokens if t in title)
                content_matches = sum(1 for t in query_tokens if t in content)
                category_matches = sum(1 for t in query_tokens if t in category)
                brand_matches = sum(1 for t in query_tokens if t in brand)

                token_ratio = (title_matches * 2.0 + content_matches + brand_matches * 1.5 + category_matches) / (len(query_tokens) * 3.0)
                score += token_ratio * 0.40

            # 3. Base RRF contribution (re-scaled)
            score += min(initial_score * 10.0, 0.10)

            # Cap between 0.0 and 1.0
            final_score = round(min(score, 1.0), 4)

            meta = dict(doc.metadata)
            meta["rerank_score"] = final_score
            scored_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=meta,
                    id=doc.id,
                )
            )

        # Sort by rerank score descending
        scored_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
        return scored_docs[:top_n]
