from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Set
import csv
import json
import os
from pathlib import Path


@dataclass
class ProductSpec:
    key: str
    value: str
    section: Optional[str] = None


@dataclass
class Product:
    product_id: str
    persian_title: str
    brand_name: Optional[str] = None
    category_name: Optional[str] = None
    model: Optional[str] = None
    image_urls: List[str] = None
    specs: List[ProductSpec] = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.specs is None:
            self.specs = []

    def get_spec_value(self, key: str) -> Optional[str]:
        for spec in self.specs:
            if spec.key == key:
                return spec.value
        return None


class Products:
    def __init__(self, products: Optional[Dict[str, Product]] = None):
        self._products: Dict[str, Product] = products or {}
        # Precomputed indexes for fast lookups
        self._by_category: Dict[str, List[Product]] = {}
        self._by_brand: Dict[str, List[Product]] = {}
        for p in self._products.values():
            if p.category_name:
                self._by_category.setdefault(p.category_name.strip(), []).append(p)
            if p.brand_name:
                self._by_brand.setdefault(p.brand_name.strip(), []).append(p)

    def get_by_id(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def get_by_category(self, category_name: str) -> List[Product]:
        return self._by_category.get(category_name.strip(), [])

    def get_by_brand(self, brand_name: str) -> List[Product]:
        return self._by_brand.get(brand_name.strip(), [])

    def filter(
        self,
        query: Optional[str] = None,
        category_name: Optional[str] = None,
        brand_name: Optional[str] = None,
        model: Optional[str] = None,
        has_images: Optional[bool] = None,
        has_specs: Optional[bool] = None,
        spec_filters: Optional[Dict[str, str]] = None,
        predicate: Optional[Callable[[Product], bool]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Product]:
        """
        Filter products based on query keywords, category, brand, specs, and pagination.
        """
        if category_name:
            candidates = self.get_by_category(category_name)
        elif brand_name:
            candidates = self.get_by_brand(brand_name)
        else:
            candidates = self._products.values()

        results = []
        q_lower = query.lower().strip() if query else None

        for p in candidates:
            if category_name and (p.category_name or "").strip() != category_name.strip():
                continue
            if brand_name and (p.brand_name or "").strip() != brand_name.strip():
                continue
            if model and (p.model or "").strip() != model.strip():
                continue
            if has_images is not None:
                if bool(p.image_urls) != has_images:
                    continue
            if has_specs is not None:
                if bool(p.specs) != has_specs:
                    continue
            if spec_filters:
                p_specs_dict = {s.key: s.value for s in p.specs}
                match_specs = True
                for sk, sv in spec_filters.items():
                    if sk not in p_specs_dict or sv.lower() not in p_specs_dict[sk].lower():
                        match_specs = False
                        break
                if not match_specs:
                    continue
            if q_lower:
                text_to_search = f"{p.persian_title} {p.brand_name or ''} {p.model or ''} {p.category_name or ''}".lower()
                if q_lower not in text_to_search:
                    continue
            if predicate and not predicate(p):
                continue

            results.append(p)

        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]

        return results

    def search_by_title(self, query: str, limit: int = 50) -> List[Product]:
        """Convenience search by title keywords."""
        return self.filter(query=query, limit=limit)

    def categories(self) -> List[str]:
        return list(self._by_category.keys())

    def brands(self) -> List[str]:
        return list(self._by_brand.keys())

    def all(self) -> List[Product]:
        return list(self._products.values())

    def __len__(self) -> int:
        return len(self._products)

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._products

    def __iter__(self):
        return iter(self._products.values())

    @classmethod
    def load_from_csv(cls, file_path: str | Path) -> "Products":
        products: Dict[str, Product] = {}
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_id = row.get("product_id")
                if not p_id:
                    continue

                def _parse_val(val: Optional[str]):
                    if val is None or val == "" or val == "null":
                        return None
                    return val

                brand_name = _parse_val(row.get("brand_name"))
                category_name = _parse_val(row.get("category_name"))
                model = _parse_val(row.get("model"))
                persian_title = _parse_val(row.get("persian_title")) or ""

                raw_images = _parse_val(row.get("image_urls"))
                image_urls: List[str] = []
                if raw_images:
                    try:
                        parsed = json.loads(raw_images)
                        if isinstance(parsed, list):
                            image_urls = parsed
                    except Exception:
                        pass

                raw_specs = _parse_val(row.get("specs"))
                specs: List[ProductSpec] = []
                if raw_specs:
                    try:
                        parsed_specs = json.loads(raw_specs)
                        if isinstance(parsed_specs, list):
                            for s in parsed_specs:
                                if isinstance(s, dict):
                                    specs.append(
                                        ProductSpec(
                                            key=s.get("key", ""),
                                            value=s.get("value", ""),
                                            section=s.get("section"),
                                        )
                                    )
                    except Exception:
                        pass

                products[p_id] = Product(
                    product_id=p_id,
                    persian_title=persian_title,
                    brand_name=brand_name,
                    category_name=category_name,
                    model=model,
                    image_urls=image_urls,
                    specs=specs,
                )
        return cls(products)
