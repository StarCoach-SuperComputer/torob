from pathlib import Path
from typing import Optional, List, Dict, Any
from .products import Products, Product
from .shops import Shops, Shop
from .offers import Offers, Offer


class Repository:
    def __init__(
        self,
        products: Optional[Products] = None,
        shops: Optional[Shops] = None,
        offers: Optional[Offers] = None,
    ):
        self.products: Products = products if products is not None else Products()
        self.shops: Shops = shops if shops is not None else Shops()
        self.offers: Offers = offers if offers is not None else Offers()

    def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Combines a product with its offers and associated shop details.
        """
        product = self.products.get_by_id(product_id)
        if not product:
            return None

        offers = self.offers.get_by_product_id(product_id)
        offers_with_shops = []
        for offer in offers:
            shop = self.shops.get_by_id(offer.shop_id)
            offers_with_shops.append({"offer": offer, "shop": shop})

        cheapest_offer = self.offers.get_cheapest_for_product(product_id)

        return {
            "product": product,
            "offers": offers_with_shops,
            "cheapest_offer": cheapest_offer,
            "offers_count": len(offers),
        }

    def get_shop_details(self, shop_id: str) -> Optional[Dict[str, Any]]:
        """
        Combines shop info with all its active offers and products.
        """
        shop = self.shops.get_by_id(shop_id)
        if not shop:
            return None

        offers = self.offers.get_by_shop_id(shop_id)
        products_in_shop = []
        for offer in offers:
            p = self.products.get_by_id(offer.product_id)
            if p:
                products_in_shop.append({"product": p, "offer": offer})

        return {
            "shop": shop,
            "offers": offers,
            "products": products_in_shop,
            "total_products": len(products_in_shop),
        }

    def find_products_in_shop(
        self, shop_id: str, query: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find products sold by a specific shop, optionally filtered by keyword.
        """
        offers = self.offers.get_by_shop_id(shop_id)
        matched = []
        q_lower = query.lower().strip() if query else None

        for offer in offers:
            product = self.products.get_by_id(offer.product_id)
            if not product:
                continue
            if q_lower:
                text = f"{product.persian_title} {product.brand_name or ''} {product.model or ''}".lower()
                if q_lower not in text:
                    continue
            matched.append({"product": product, "offer": offer})
            if len(matched) >= limit:
                break
        return matched

    def find_shops_selling_product(
        self,
        product_id: str,
        near_lat: Optional[float] = None,
        near_lon: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        supports_fast_delivery: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all shops selling a specific product, with optional location and delivery filters.
        """
        offers = self.offers.get_by_product_id(product_id)
        results = []

        for offer in offers:
            shop = self.shops.get_by_id(offer.shop_id)
            if not shop:
                continue
            if supports_fast_delivery is not None and shop.supports_fast_delivery != supports_fast_delivery:
                continue
            dist = None
            if near_lat is not None and near_lon is not None:
                dist = shop.distance_to(near_lat, near_lon)
                if max_distance_km is not None and (dist is None or dist > max_distance_km):
                    continue

            results.append({"shop": shop, "offer": offer, "distance_km": dist})

        if near_lat is not None and near_lon is not None:
            results.sort(
                key=lambda x: x["distance_km"] if x["distance_km"] is not None else float("inf")
            )

        return results

    @classmethod
    def load_from_directory(cls, data_dir: str | Path) -> "Repository":
        data_path = Path(data_dir)
        products_file = data_path / "products.csv"
        shops_file = data_path / "shops.csv"
        offers_file = data_path / "offers.csv"

        products = (
            Products.load_from_csv(products_file)
            if products_file.exists()
            else Products()
        )
        shops = (
            Shops.load_from_csv(shops_file)
            if shops_file.exists()
            else Shops()
        )
        offers = (
            Offers.load_from_csv(offers_file)
            if offers_file.exists()
            else Offers()
        )

        return cls(products=products, shops=shops, offers=offers)
