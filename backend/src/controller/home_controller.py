import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query, HTTPException, Request

from rag.embedding import Embedding
from rag.qdrant_store import QdrantProductStore
from rag.retriever import HybridRetriever
from rag.agent import ShoppingAgent, CustomerProfile

router = APIRouter(prefix="/search", tags=["Search & Retrieval"])

# Global singleton storage, retriever & shopping agent
_store: Optional[QdrantProductStore] = None
_retriever: Optional[HybridRetriever] = None
_agent: Optional[ShoppingAgent] = None


def get_agent(repo=None) -> ShoppingAgent:
    global _agent
    if _agent is None:
        retriever = get_retriever()
        _agent = ShoppingAgent(retriever=retriever, repository=repo)
    elif repo and _agent.repository is None:
        _agent.repository = repo
    return _agent


def get_retriever() -> HybridRetriever:
    """
    Lazy singleton initialization for QdrantProductStore and HybridRetriever.
    Can be configured via environment variables QDRANT_URL or QDRANT_STORAGE_PATH.
    Defaults to http://localhost:6333 if available or running via docker.
    """
    global _store, _retriever
    if _retriever is None:
        qdrant_url = os.getenv("QDRANT_URL")
        storage_path = os.getenv("QDRANT_STORAGE_PATH")
        collection_name = os.getenv("QDRANT_COLLECTION", "products")

        # If neither QDRANT_URL nor QDRANT_STORAGE_PATH is explicitly set,
        # default to http://localhost:6333 (Docker Qdrant) first
        if not qdrant_url and not storage_path:
            qdrant_url = "http://localhost:6333"

        embedding = Embedding()
        _store = QdrantProductStore(
            embedding=embedding,
            url=qdrant_url,
            storage_path=storage_path if not qdrant_url else None,
            collection_name=collection_name,
        )
        _retriever = HybridRetriever(store=_store)
    return _retriever


class SearchResultItem(BaseModel):
    product_id: str
    chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None
    persian_title: Optional[str] = None
    brand_name: Optional[str] = None
    category_name: Optional[str] = None
    model: Optional[str] = None
    score: float = Field(..., description="Precision relevance score (from Reranker if active, or normalized RRF)")
    raw_rrf_score: Optional[float] = Field(default=None, description="Raw reciprocal rank fusion score")
    rerank_score: Optional[float] = Field(default=None, description="Score assigned by cross-encoder reranker")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    product_details: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    limit: int = Field(default=10, ge=1, le=50, description="Max results to return")
    group_by_product: bool = Field(
        default=True,
        description="Whether to deduplicate multiple chunks of the same product",
    )
    chunk_type: Optional[str] = Field(
        default=None,
        description="Filter by chunk type ('summary' or 'specs')",
    )
    category_name: Optional[str] = Field(
        default=None,
        description="Optional filter by exact category name",
    )
    brand_name: Optional[str] = Field(
        default=None,
        description="Optional filter by brand name",
    )
    use_reranker: bool = Field(
        default=True,
        description="Apply precision Cross-Encoder reranking on retrieved candidates",
    )
    with_details: bool = Field(
        default=False,
        description="If True, populates product offers, price, and shops from repository",
    )


@router.post("", response_model=SearchResponse)
async def search_products(request_body: SearchRequest, req: Request):
    """
    Hybrid semantic + keyword search endpoint using Qdrant and Reciprocal Rank Fusion (RRF).
    """
    retriever = get_retriever()

    # Build metadata filters
    filter_dict = {}
    if request_body.category_name:
        filter_dict["category_name"] = request_body.category_name
    if request_body.brand_name:
        filter_dict["brand_name"] = request_body.brand_name

    try:
        docs = retriever.retrieve(
            query=request_body.query,
            limit=request_body.limit,
            chunk_type=request_body.chunk_type,
            filter_dict=filter_dict if filter_dict else None,
            group_by_product=request_body.group_by_product,
            use_reranker=request_body.use_reranker,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search retrieval error: {str(e)}")

    # Access main app repository if with_details is requested
    repo = getattr(req.app.state, "repo", None)

    # Maximum theoretical RRF score when an item is rank #1 in both channels (1/(60+1) + 1/(60+1))
    max_rrf_possible = (1.0 / (retriever.rrf_k + 1)) * (retriever.dense_weight + retriever.sparse_weight)

    items: List[SearchResultItem] = []
    for doc in docs:
        meta = doc.metadata
        product_id = meta.get("product_id", "")
        raw_rrf = meta.get("rrf_score")
        rerank_sc = meta.get("rerank_score")

        # Determine primary final score: Rerank score if available, otherwise normalized RRF
        if rerank_sc is not None:
            final_score = round(float(rerank_sc), 4)
        elif raw_rrf is not None:
            final_score = round(min(raw_rrf / max_rrf_possible, 1.0), 4)
        else:
            final_score = round(float(meta.get("score", 0.0)), 4)

        details = None
        if request_body.with_details and repo and product_id:
            details = repo.get_product_details(product_id)

        items.append(
            SearchResultItem(
                product_id=product_id,
                chunk_id=meta.get("chunk_id", doc.id),
                chunk_type=meta.get("chunk_type"),
                persian_title=meta.get("persian_title"),
                brand_name=meta.get("brand_name"),
                category_name=meta.get("category_name"),
                model=meta.get("model"),
                score=final_score,
                raw_rrf_score=round(raw_rrf, 6) if raw_rrf is not None else None,
                rerank_score=round(rerank_sc, 4) if rerank_sc is not None else None,
                content=doc.page_content,
                metadata={k: v for k, v in meta.items() if k not in ("rrf_score", "score", "rerank_score")},
                product_details=details,
            )
        )

    return SearchResponse(
        query=request_body.query,
        total_results=len(items),
        results=items,
    )


@router.get("", response_model=SearchResponse)
async def search_products_get(
    req: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    group_by_product: bool = Query(True),
    chunk_type: Optional[str] = Query(None),
    category_name: Optional[str] = Query(None),
    brand_name: Optional[str] = Query(None),
    use_reranker: bool = Query(True),
    with_details: bool = Query(False),
):
    """
    Convenience GET endpoint for query parameter searches.
    """
    search_req = SearchRequest(
        query=q,
        limit=limit,
        group_by_product=group_by_product,
        chunk_type=chunk_type,
        category_name=category_name,
        brand_name=brand_name,
        use_reranker=use_reranker,
        with_details=with_details,
    )
    return await search_products(search_req, req)


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer chat message")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuous conversation memory")
    customer_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Visual / demographic data (age_group, gender, style_preference, clothing_colors_detected)",
    )


class AgentChatResponse(BaseModel):
    session_id: str
    reply: str
    search_intent: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Search intent and query formulated by the agent prior to database retrieval",
    )
    retrieved_products: List[Dict[str, Any]]
    cross_sell_offer: Optional[Dict[str, Any]] = None
    customer_profile: Dict[str, Any]
    history_length: int


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req_body: AgentChatRequest, req: Request):
    """
    Conversational Shopping Assistant endpoint with RAG grounding,
    vision customer profile integration, memory, and cross-sell bundle recommendations.
    """
    repo = getattr(req.app.state, "repo", None)
    agent = get_agent(repo=repo)

    profile = None
    if req_body.customer_profile:
        profile = CustomerProfile(
            age_group=req_body.customer_profile.get("age_group"),
            gender=req_body.customer_profile.get("gender"),
            style_preference=req_body.customer_profile.get("style_preference"),
            mood_or_expression=req_body.customer_profile.get("mood_or_expression"),
            clothing_colors_detected=req_body.customer_profile.get("clothing_colors_detected", []),
            budget_tier=req_body.customer_profile.get("budget_tier"),
            extra_attributes=req_body.customer_profile.get("extra_attributes", {}),
        )

    try:
        response_data = agent.chat(
            user_message=req_body.message,
            session_id=req_body.session_id,
            customer_profile=profile,
        )
        return AgentChatResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

