from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import csv
import os
from pathlib import Path


@dataclass
class Offer:
    offer_id: str
    product_id: str
    shop_id: str
    price_toman: int
    seller_title: Optional[str] = None
    description: Optional[str] = None
    last_price_changed_at: Optional[str] = None
    warranty: Optional[str] = None


class Offers:
    def __init__(self, offers: Optional[Dict[str, Offer]] = None):
        self._offers: Dict[str, Offer] = offers or {}
        # Secondary indexes for fast lookup
        self._by_product_id: Dict[str, List[Offer]] = {}
        self._by_shop_id: Dict[str, List[Offer]] = {}
        for offer in self._offers.values():
            self._by_product_id.setdefault(offer.product_id, []).append(offer)
            self._by_shop_id.setdefault(offer.shop_id, []).append(offer)

    def get_by_id(self, offer_id: str) -> Optional[Offer]:
        return self._offers.get(offer_id)

    def get_by_product_id(self, product_id: str) -> List[Offer]:
        return self._by_product_id.get(product_id, [])

    def get_by_shop_id(self, shop_id: str) -> List[Offer]:
        return self._by_shop_id.get(shop_id, [])

    def get_cheapest_for_product(self, product_id: str) -> Optional[Offer]:
        """Returns the offer with the lowest price_toman > 0 for a product."""
        offers = self.get_by_product_id(product_id)
        valid_offers = [o for o in offers if o.price_toman > 0]
        if not valid_offers:
            return offers[0] if offers else None
        return min(valid_offers, key=lambda o: o.price_toman)

    def filter(
        self,
        product_id: Optional[str] = None,
        shop_id: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        has_warranty: Optional[bool] = None,
        warranty_query: Optional[str] = None,
        seller_title_query: Optional[str] = None,
        sort_by_price: Optional[str] = None,  # 'asc' or 'desc'
        predicate: Optional[Callable[[Offer], bool]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Offer]:
        """
        Filter offers by product, shop, price range, warranty, and seller title.
        """
        if product_id:
            candidates = self.get_by_product_id(product_id)
        elif shop_id:
            candidates = self.get_by_shop_id(shop_id)
        else:
            candidates = self._offers.values()

        results = []
        w_lower = warranty_query.lower().strip() if warranty_query else None
        s_lower = seller_title_query.lower().strip() if seller_title_query else None

        for o in candidates:
            if product_id and o.product_id != product_id:
                continue
            if shop_id and o.shop_id != shop_id:
                continue
            if min_price is not None and o.price_toman < min_price:
                continue
            if max_price is not None and o.price_toman > max_price:
                continue
            if has_warranty is not None:
                has_w = bool(o.warranty and o.warranty.strip())
                if has_w != has_warranty:
                    continue
            if w_lower:
                if not o.warranty or w_lower not in o.warranty.lower():
                    continue
            if s_lower:
                if not o.seller_title or s_lower not in o.seller_title.lower():
                    continue
            if predicate and not predicate(o):
                continue

            results.append(o)

        if sort_by_price == "asc":
            results.sort(key=lambda o: o.price_toman)
        elif sort_by_price == "desc":
            results.sort(key=lambda o: o.price_toman, reverse=True)

        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]

        return results

    def all(self) -> List[Offer]:
        return list(self._offers.values())

    def __len__(self) -> int:
        return len(self._offers)

    def __contains__(self, offer_id: str) -> bool:
        return offer_id in self._offers

    def __iter__(self):
        return iter(self._offers.values())

    @classmethod
    def load_from_csv(cls, file_path: str | Path) -> "Offers":
        offers: Dict[str, Offer] = {}
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                o_id = row.get("offer_id")
                if not o_id:
                    continue

                def _parse_val(val: Optional[str]):
                    if val is None or val == "" or val == "null":
                        return None
                    return val

                product_id = _parse_val(row.get("product_id")) or ""
                shop_id = _parse_val(row.get("shop_id")) or ""
                seller_title = _parse_val(row.get("seller_title"))
                description = _parse_val(row.get("description"))
                last_price_changed_at = _parse_val(row.get("last_price_changed_at"))
                warranty = _parse_val(row.get("warranty"))

                price_raw = _parse_val(row.get("price_toman"))
                price_toman = 0
                if price_raw is not None:
                    try:
                        price_toman = int(float(price_raw))
                    except ValueError:
                        price_toman = 0

                offers[o_id] = Offer(
                    offer_id=o_id,
                    product_id=product_id,
                    shop_id=shop_id,
                    price_toman=price_toman,
                    seller_title=seller_title,
                    description=description,
                    last_price_changed_at=last_price_changed_at,
                    warranty=warranty,
                )
        return cls(offers)
