from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import csv
import json
import math
import os
from pathlib import Path


@dataclass
class ShopContact:
    channel: str
    display_number: Optional[str] = None


@dataclass
class WeeklyHours:
    raw: Dict[str, Any]

    def is_open_on(self, day_abbr: str) -> bool:
        """e.g. day_abbr: 'sat', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri'"""
        day_info = self.raw.get(day_abbr.lower(), {})
        return day_info.get("status") == "declared"


@dataclass
class Shop:
    shop_id: str
    name: str
    address: Optional[str] = None
    category_names: List[str] = None
    contacts: List[ShopContact] = None
    instagram: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_label: Optional[str] = None
    payment_options: List[str] = None
    supports_fast_delivery: Optional[bool] = None
    website_url: Optional[str] = None
    weekly_hours: Optional[WeeklyHours] = None

    def __post_init__(self):
        if self.category_names is None:
            self.category_names = []
        if self.contacts is None:
            self.contacts = []
        if self.payment_options is None:
            self.payment_options = []

    def distance_to(self, lat: float, lon: float) -> Optional[float]:
        """Calculates distance in kilometers using the Haversine formula."""
        if self.lat is None or self.lon is None:
            return None
        r = 6371.0  # Earth's radius in km
        d_lat = math.radians(self.lat - lat)
        d_lon = math.radians(self.lon - lon)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat))
            * math.cos(math.radians(self.lat))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c


class Shops:
    def __init__(self, shops: Optional[Dict[str, Shop]] = None):
        self._shops: Dict[str, Shop] = shops or {}
        # Precomputed index by category and location_label
        self._by_category: Dict[str, List[Shop]] = {}
        self._by_location: Dict[str, List[Shop]] = {}
        for s in self._shops.values():
            if s.location_label:
                self._by_location.setdefault(s.location_label.strip(), []).append(s)
            for cat in s.category_names:
                self._by_category.setdefault(cat.strip(), []).append(s)

    def get_by_id(self, shop_id: str) -> Optional[Shop]:
        return self._shops.get(shop_id)

    def get_by_location(self, location_label: str) -> List[Shop]:
        return self._by_location.get(location_label.strip(), [])

    def get_by_category(self, category_name: str) -> List[Shop]:
        return self._by_category.get(category_name.strip(), [])

    def filter(
        self,
        query: Optional[str] = None,
        location_label: Optional[str] = None,
        category_name: Optional[str] = None,
        supports_fast_delivery: Optional[bool] = None,
        has_website: Optional[bool] = None,
        has_instagram: Optional[bool] = None,
        payment_option: Optional[str] = None,
        near_lat: Optional[float] = None,
        near_lon: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        predicate: Optional[Callable[[Shop], bool]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Shop]:
        """
        Filter shops by keyword, location, categories, delivery, payment options, and geo distance.
        """
        if location_label and location_label.strip() in self._by_location:
            candidates = self.get_by_location(location_label)
        elif category_name and category_name.strip() in self._by_category:
            candidates = self.get_by_category(category_name)
        else:
            candidates = self._shops.values()

        results = []
        q_lower = query.lower().strip() if query else None

        for s in candidates:
            if location_label and (s.location_label or "").strip() != location_label.strip():
                continue
            if category_name and not any(cat.strip() == category_name.strip() for cat in s.category_names):
                continue
            if supports_fast_delivery is not None and s.supports_fast_delivery != supports_fast_delivery:
                continue
            if has_website is not None and bool(s.website_url) != has_website:
                continue
            if has_instagram is not None and bool(s.instagram) != has_instagram:
                continue
            if payment_option and payment_option not in s.payment_options:
                continue
            if near_lat is not None and near_lon is not None and max_distance_km is not None:
                dist = s.distance_to(near_lat, near_lon)
                if dist is None or dist > max_distance_km:
                    continue
            if q_lower:
                text_to_search = f"{s.name} {s.address or ''} {s.location_label or ''}".lower()
                if q_lower not in text_to_search:
                    continue
            if predicate and not predicate(s):
                continue

            results.append(s)

        # Sort by distance if geo coordinates are provided
        if near_lat is not None and near_lon is not None:
            results.sort(
                key=lambda s: s.distance_to(near_lat, near_lon)
                if s.distance_to(near_lat, near_lon) is not None
                else float("inf")
            )

        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]

        return results

    def search_by_name(self, query: str, limit: int = 50) -> List[Shop]:
        return self.filter(query=query, limit=limit)

    def locations(self) -> List[str]:
        return list(self._by_location.keys())

    def categories(self) -> List[str]:
        return list(self._by_category.keys())

    def all(self) -> List[Shop]:
        return list(self._shops.values())

    def __len__(self) -> int:
        return len(self._shops)

    def __contains__(self, shop_id: str) -> bool:
        return shop_id in self._shops

    def __iter__(self):
        return iter(self._shops.values())

    @classmethod
    def load_from_csv(cls, file_path: str | Path) -> "Shops":
        shops: Dict[str, Shop] = {}
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_id = row.get("shop_id")
                if not s_id:
                    continue

                def _parse_val(val: Optional[str]):
                    if val is None or val == "" or val == "null":
                        return None
                    return val

                address = _parse_val(row.get("address"))
                name = _parse_val(row.get("name")) or ""
                instagram = _parse_val(row.get("instagram"))
                location_label = _parse_val(row.get("location_label"))
                website_url = _parse_val(row.get("website_url"))

                lat_raw = _parse_val(row.get("lat"))
                lat: Optional[float] = None
                if lat_raw is not None:
                    try:
                        lat = float(lat_raw)
                    except ValueError:
                        pass

                lon_raw = _parse_val(row.get("lon"))
                lon: Optional[float] = None
                if lon_raw is not None:
                    try:
                        lon = float(lon_raw)
                    except ValueError:
                        pass

                fast_delivery_raw = _parse_val(row.get("supports_fast_delivery"))
                supports_fast_delivery: Optional[bool] = None
                if fast_delivery_raw is not None:
                    supports_fast_delivery = fast_delivery_raw.lower() in ("true", "1", "yes")

                category_names_raw = _parse_val(row.get("category_names"))
                category_names: List[str] = []
                if category_names_raw:
                    try:
                        parsed = json.loads(category_names_raw)
                        if isinstance(parsed, list):
                            category_names = [str(item) for item in parsed]
                    except Exception:
                        pass

                contacts_raw = _parse_val(row.get("contacts"))
                contacts: List[ShopContact] = []
                if contacts_raw:
                    try:
                        parsed_contacts = json.loads(contacts_raw)
                        if isinstance(parsed_contacts, list):
                            for c in parsed_contacts:
                                if isinstance(c, dict):
                                    contacts.append(
                                        ShopContact(
                                            channel=c.get("channel", ""),
                                            display_number=c.get("display_number"),
                                        )
                                    )
                    except Exception:
                        pass

                payment_options_raw = _parse_val(row.get("payment_options"))
                payment_options: List[str] = []
                if payment_options_raw:
                    try:
                        parsed_po = json.loads(payment_options_raw)
                        if isinstance(parsed_po, list):
                            payment_options = [str(item) for item in parsed_po]
                    except Exception:
                        pass

                weekly_hours_raw = _parse_val(row.get("weekly_hours"))
                weekly_hours: Optional[WeeklyHours] = None
                if weekly_hours_raw:
                    try:
                        parsed_wh = json.loads(weekly_hours_raw)
                        if isinstance(parsed_wh, dict):
                            weekly_hours = WeeklyHours(raw=parsed_wh)
                    except Exception:
                        pass

                shops[s_id] = Shop(
                    shop_id=s_id,
                    name=name,
                    address=address,
                    category_names=category_names,
                    contacts=contacts,
                    instagram=instagram,
                    lat=lat,
                    lon=lon,
                    location_label=location_label,
                    payment_options=payment_options,
                    supports_fast_delivery=supports_fast_delivery,
                    website_url=website_url,
                    weekly_hours=weekly_hours,
                )
        return cls(shops)
