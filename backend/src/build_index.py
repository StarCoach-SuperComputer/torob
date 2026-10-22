import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Ensure localhost traffic does not get routed through SOCKS proxy
no_proxy = os.environ.get("no_proxy", "")
localhost_entries = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = f"{no_proxy},{localhost_entries}" if no_proxy else localhost_entries
os.environ["NO_PROXY"] = os.environ["no_proxy"]

# Ensure backend/src is on sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from repository.products import Products
from rag.embedding import Embedding
from rag.qdrant_store import QdrantProductStore


def find_default_data_path() -> Path:
    """Finds products.csv in standard locations."""
    candidates = [
        Path.cwd() / "data" / "products.csv",
        Path.cwd().parent / "data" / "products.csv",
        Path(__file__).resolve().parent.parent.parent / "data" / "products.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def find_default_storage_path() -> Path:
    """Finds default Qdrant storage location."""
    return Path.cwd() / "qdrant_storage"


def build_index(
    products_csv: Path,
    storage_path: Optional[Path] = None,
    url: Optional[str] = None,
    collection_name: str = "products",
    batch_size: int = 64,
    limit: Optional[int] = None,
    recreate: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
):
    print("=" * 60)
    print("📦 Qdrant Product Embedding & Ingestion CLI")
    print("=" * 60)

    if not products_csv.exists():
        print(f"❌ Error: products.csv not found at: {products_csv}")
        sys.exit(1)

    print(f"• Products file:   {products_csv.resolve()}")
    if url:
        print(f"• Target Qdrant:   {url} (Server mode)")
    else:
        print(f"• Storage folder:  {storage_path.resolve() if storage_path else 'qdrant_storage'} (Disk mode)")
    print(f"• Collection name: {collection_name}")
    print(f"• Model:           {model_name}")
    print(f"• Batch size:      {batch_size}")
    if limit:
        print(f"• Limit:           {limit} products")
    print("-" * 60)

    # 1. Load products from CSV
    print(f"⏳ Loading products from {products_csv.name}...")
    t0 = time.time()
    products_repo = Products.load_from_csv(products_csv)
    products_list = products_repo.all()
    print(f"✅ Loaded {len(products_list):,} products in {time.time() - t0:.2f}s")

    if limit and limit > 0:
        products_list = products_list[:limit]
        print(f"🔹 Trimmed to {len(products_list):,} products according to --limit")

    if not products_list:
        print("⚠️ No products found to index.")
        return

    # 2. Initialize Embedding & Qdrant store
    embedding = Embedding(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    if not embedding.api_key:
        print("⚠️ Warning: BOXAI_API_KEY / OPENAI_API_KEY is not set!")
        print("   Zero-vector embeddings will be generated for fallback/offline testing.")

    store = QdrantProductStore(
        embedding=embedding,
        url=url,
        storage_path=storage_path,
        collection_name=collection_name,
    )

    if recreate:
        print(f"🗑️ Recreating collection '{collection_name}'...")
        try:
            store.client.delete_collection(collection_name)
        except Exception:
            pass
        store._ensure_collection()

    # 3. Chunking & Embedding
    print(f"🚀 Chunking & indexing {len(products_list):,} products into Qdrant...")
    t_start = time.time()
    total_chunks = 0
    total_batches = (len(products_list) + batch_size - 1) // batch_size

    for i in range(0, len(products_list), batch_size):
        batch = products_list[i : i + batch_size]
        batch_idx = (i // batch_size) + 1

        # Collect chunks for this batch
        batch_chunks = []
        for p in batch:
            batch_chunks.extend(Embedding.chunk(p))

        added = store.add_chunks(batch_chunks, batch_size=batch_size)
        total_chunks += added

        progress = (min(i + batch_size, len(products_list)) / len(products_list)) * 100
        print(
            f"  [{batch_idx}/{total_batches}] Indexed {min(i + batch_size, len(products_list)):,}/{len(products_list):,} products "
            f"({total_chunks:,} chunks total) - {progress:.1f}%"
        )

    duration = time.time() - t_start
    print("-" * 60)
    print("🎉 Ingestion complete!")
    print(f"• Total products processed: {len(products_list):,}")
    print(f"• Total chunks in Qdrant:    {total_chunks:,}")
    print(f"• Total collection count:    {store.count():,}")
    print(f"• Elapsed time:              {duration:.2f}s ({total_chunks / max(duration, 0.001):.1f} chunks/sec)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Index and store product embeddings into local on-disk Qdrant database."
    )
    parser.add_argument(
        "--data-file",
        "-d",
        type=Path,
        default=find_default_data_path(),
        help="Path to products.csv (default: data/products.csv or ../data/products.csv)",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default=os.getenv("QDRANT_URL"),
        help="Qdrant server URL (e.g. http://localhost:6333). If not provided, uses local on-disk storage.",
    )
    parser.add_argument(
        "--storage-path",
        "-s",
        type=Path,
        default=find_default_storage_path(),
        help="Path to local Qdrant storage directory (default: ./qdrant_storage)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="products",
        help="Qdrant collection name (default: products)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size for embedding requests and DB upserts (default: 64)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of products to process (useful for testing)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection from scratch (discards existing vectors)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (defaults to BOXAI_API_KEY or OPENAI_API_KEY env vars)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for embeddings endpoint (defaults to BOXAI_BASE_URL or https://ai.boxapi.ir/ai/v1)",
    )

    args = parser.parse_args()

    build_index(
        products_csv=args.data_file,
        storage_path=args.storage_path,
        url=args.url,
        collection_name=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
