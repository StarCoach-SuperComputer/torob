import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from repository import Repository

def test_load():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    print(f"Loading data from {data_dir}...")
    repo = Repository.load_from_directory(data_dir)
    print(f"Loaded {len(repo.shops)} shops.")
    print(f"Loaded {len(repo.products)} products.")
    print(f"Loaded {len(repo.offers)} offers.")

    # Sample tests
    sample_shop = next(iter(repo.shops), None)
    if sample_shop:
        print(f"Sample shop: {sample_shop.name} ({sample_shop.shop_id}), contacts: {sample_shop.contacts}")

    sample_product = next(iter(repo.products), None)
    if sample_product:
        print(f"Sample product: {sample_product.persian_title} ({sample_product.product_id}), specs count: {len(sample_product.specs)}")

    sample_offer = next(iter(repo.offers), None)
    if sample_offer:
        print(f"Sample offer: {sample_offer.offer_id}, price: {sample_offer.price_toman}, product_id: {sample_offer.product_id}")
        offers_for_prod = repo.offers.get_by_product_id(sample_offer.product_id)
        print(f"Offers found for product {sample_offer.product_id}: {len(offers_for_prod)}")

if __name__ == "__main__":
    test_load()
