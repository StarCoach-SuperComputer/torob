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

    def get_product_details_dict(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns a JSON-serializable dictionary combining product, real shops from Shops,
        and real offers from Offers.
        """
        product = self.products.get_by_id(product_id)
        offers = self.offers.get_by_product_id(product_id)
        cheapest_offer = self.offers.get_cheapest_for_product(product_id)

        if not product and not offers:
            return None

        # Build shops list with joined Shop details
        shops_list = []
        for offer in offers:
            shop = self.shops.get_by_id(offer.shop_id)
            shop_name = shop.name if shop else (offer.seller_title or "فروشگاه همکار ترب")
            city = ""
            if shop and shop.location_label:
                city = shop.location_label
            elif shop and shop.address:
                city = shop.address[:40]
            else:
                city = "تهران"

            shops_list.append({
                "offer_id": offer.offer_id,
                "shop_id": offer.shop_id,
                "shop_name": shop_name,
                "price": offer.price_toman,
                "price_formatted": f"{offer.price_toman:,} تومان" if offer.price_toman > 0 else "استعلام قیمت",
                "city": city,
                "address": shop.address if shop else None,
                "warranty": offer.warranty or "گارانتی اصالت و سلامت فیزیکی کالا",
                "supports_fast_delivery": bool(shop.supports_fast_delivery) if shop else False,
                "website_url": shop.website_url if shop else None,
                "instagram": shop.instagram if shop else None,
                "seller_title": offer.seller_title,
                "description": offer.description,
            })

        # Sort shops by price (cheapest first)
        shops_list.sort(key=lambda s: s["price"] if s["price"] > 0 else float("inf"))

        # Fallback if product has no direct offers in offers.csv:
        # Match real shops from Shops by category or catalog
        if not shops_list:
            cat_name = product.category_name if product else None
            cat_shops = self.shops.get_by_category(cat_name) if cat_name else []
            if not cat_shops:
                cat_shops = self.shops.all()[:5]

            for s in cat_shops[:5]:
                city = s.location_label or (s.address[:35] if s.address else "تهران، ارسال فوری")
                shops_list.append({
                    "offer_id": f"offer_{s.shop_id}",
                    "shop_id": s.shop_id,
                    "shop_name": s.name,
                    "price": 0,
                    "price_formatted": "استعلام قیمت روز",
                    "city": city,
                    "address": s.address,
                    "warranty": "ضمانت ۷ روز بازگشت وجه ترب",
                    "supports_fast_delivery": bool(s.supports_fast_delivery),
                    "website_url": s.website_url,
                    "instagram": s.instagram,
                    "seller_title": s.name,
                    "description": None,
                })

        # Calculate lowest price
        min_price = None
        if cheapest_offer and cheapest_offer.price_toman > 0:
            min_price = cheapest_offer.price_toman
        elif shops_list and shops_list[0]["price"] > 0:
            min_price = shops_list[0]["price"]

        min_price_str = f"از {min_price:,} تومان" if min_price else "استعلام در فروشگاه‌ها"

        # Image URL
        image_urls = product.image_urls if product else []
        primary_image = image_urls[0] if image_urls else None

        # Specs
        specs_list = []
        if product and product.specs:
            specs_list = [{"key": s.key, "value": s.value, "section": s.section} for s in product.specs]

        title = product.persian_title if product else (offers[0].seller_title if offers else "محصول ترب")

        return {
            "product_id": product_id,
            "persian_title": title,
            "brand_name": product.brand_name if product else None,
            "category_name": product.category_name if product else None,
            "model": product.model if product else None,
            "image_urls": image_urls,
            "primary_image": primary_image,
            "specs": specs_list,
            "min_price": min_price,
            "min_price_formatted": min_price_str,
            "offers_count": len(shops_list),
            "shops": shops_list,
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
