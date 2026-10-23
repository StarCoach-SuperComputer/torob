import os
import json
import base64
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import httpx
from dotenv import load_dotenv

from .agent import CustomerProfile

# Ensure .env is loaded
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("torob.vision")

VISION_PERSONA_PROMPT = """You are an expert fashion stylist, visual demographer, and AI persona profiler for Torob (an Iranian e-commerce platform).
Analyze the provided customer image and extract a structured visual profile.

Pay close attention to:
1. is_young / age_group: approximate age classification (e.g. "young_adult", "teen", "middle_aged", "senior") and approximate age range (e.g. "20-28").
2. gender: perceived gender ("male", "female", "unisex").
3. current_clothing: detailed description of what the person is wearing (e.g., "black hoodie and blue jeans", "formal navy suit with white shirt").
4. clothing_colors: array of prominent detected clothing colors in Persian (e.g. ["مشکی", "آبی", "سفید", "طوسی", "قرمز", "کرم", "سبز", "قهوه‌ای", "زرد"]).
5. style_preference: overall fashion style (e.g. "اسپرت / راحتی (casual/sport)", "رسمی / کلاسیک (formal)", "استریت‌ویر (streetwear)", "مینیمال (minimalist)").
6. mood_or_expression: mood or expression (e.g. "شاد / پرانرژی", "آرام / متفکر", "عجول / بیزنس").
7. budget_tier: estimated budget preference tier based on accessories/style ("budget", "mid-range", "premium").
8. extra_attributes: other useful styling observations such as season ("تابستان", "زمستان"), body type, hair style, or specific accessories (watch, glasses, hat, bag).

You MUST respond strictly with valid JSON without any markdown formatting or surrounding text:
{
    "age_group": "young_adult (20-25)",
    "is_young": true,
    "gender": "male",
    "current_clothing": "تی‌شرت مشکی یقه گرد و شلوار جین سرمه‌ای",
    "clothing_colors": ["مشکی", "سرمه‌ای"],
    "style_preference": "اسپرت",
    "mood_or_expression": "خندان و پرانرژی",
    "budget_tier": "mid-range",
    "extra_attributes": {
        "season": "تابستان",
        "has_glasses": false,
        "accessories": ["ساعت مچی اسپرت"]
    }
}
"""


class VisionPersonaExtractor:
    """
    Extracts customer persona (demographics, clothing, colors, style) from images
    using state-of-the-art vision models (e.g. GPT-4o / GPT-4o-mini / BoxAI vision).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("BOXAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("BOXAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://ai.boxapi.ir/ai/v1"
        ).rstrip("/")

    def extract_from_base64_or_url(
        self,
        image_data: str,
        customer_id: Optional[str] = None,
    ) -> CustomerProfile:
        """
        Takes a base64 data URI (data:image/jpeg;base64,...) or raw base64 or an HTTP/HTTPS image URL
        and returns a populated CustomerProfile object.
        """
        # Format the image URL for OpenAI / BoxAI vision schema
        if image_data.startswith("http://") or image_data.startswith("https://"):
            image_url_obj = {"url": image_data}
        elif image_data.startswith("data:image/"):
            image_url_obj = {"url": image_data}
        else:
            # Assume raw base64 string, default to jpeg
            image_url_obj = {"url": f"data:image/jpeg;base64,{image_data}"}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PERSONA_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": image_url_obj,
                    },
                ],
            }
        ]

        if not self.api_key:
            logger.warning("No API key available for Vision model. Returning default profile.")
            return self._fallback_profile(customer_id)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()
                raw_text = result["choices"][0]["message"]["content"].strip()

                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    raw_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

                parsed = json.loads(raw_text)

                extra_attrs = parsed.get("extra_attributes", {})
                if "current_clothing" in parsed:
                    extra_attrs["current_clothing"] = parsed["current_clothing"]
                if "is_young" in parsed:
                    extra_attrs["is_young"] = parsed["is_young"]

                return CustomerProfile(
                    customer_id=customer_id,
                    age_group=parsed.get("age_group"),
                    gender=parsed.get("gender"),
                    style_preference=parsed.get("style_preference"),
                    mood_or_expression=parsed.get("mood_or_expression"),
                    clothing_colors_detected=parsed.get("clothing_colors", []),
                    budget_tier=parsed.get("budget_tier"),
                    extra_attributes=extra_attrs,
                )
        except Exception as e:
            logger.error(f"Vision API extraction failed: {e}")
            return self._fallback_profile(customer_id)

    def extract_from_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        customer_id: Optional[str] = None,
    ) -> CustomerProfile:
        """
        Takes raw image bytes, encodes to base64 data URI and invokes extraction.
        """
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_str}"
        return self.extract_from_base64_or_url(data_uri, customer_id=customer_id)

    def _fallback_profile(self, customer_id: Optional[str] = None) -> CustomerProfile:
        return CustomerProfile(
            customer_id=customer_id,
            age_group="جوان (۲۰-۳۰ سال)",
            gender="نامشخص",
            style_preference="اسپرت / کژوال",
            mood_or_expression="عادی",
            clothing_colors_detected=["مشکی"],
            budget_tier="mid-range",
            extra_attributes={"fallback": True, "current_clothing": "لباس اسپرت"},
        )
