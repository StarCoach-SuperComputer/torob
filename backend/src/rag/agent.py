import os
import json
import logging
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("torob.agent")
from langchain_core.documents import Document
from .retriever import HybridRetriever
from repository import Repository


@dataclass
class CustomerProfile:
    """
    Customer demographic and contextual profile,
    typically extracted from a camera/vision model analysis or past history.
    """
    customer_id: Optional[str] = None
    age_group: Optional[str] = None       # e.g., "young_adult", "teen", "middle_aged", "senior" or estimated age (e.g. "25-30")
    gender: Optional[str] = None          # e.g., "male", "female", "unisex"
    style_preference: Optional[str] = None # e.g., "casual", "formal", "sport", "streetwear", "minimalist"
    mood_or_expression: Optional[str] = None # e.g., "happy", "hurried", "thoughtful"
    clothing_colors_detected: List[str] = field(default_factory=list) # e.g., ["black", "navy", "white"]
    budget_tier: Optional[str] = None     # e.g., "budget", "mid-range", "premium"
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_persona_summary(self) -> str:
        lines = []
        if self.gender:
            lines.append(f"جنسیت تقریبی: {self.gender}")
        if self.age_group:
            lines.append(f"رده سنی: {self.age_group}")
        if self.style_preference:
            lines.append(f"سبک پوشش و استایل: {self.style_preference}")
        if self.clothing_colors_detected:
            lines.append(f"رنگ‌های لباس یا آیتم‌های فعلی: {', '.join(self.clothing_colors_detected)}")
        if self.budget_tier:
            lines.append(f"سطح بودجه: {self.budget_tier}")
        if not lines:
            return "اطلاعات بصری خاصی از چهره یا ظاهر کاربر ثبت نشده است."
        return "\n".join(lines)


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchIntent:
    """
    Search intent formulated by the agent prior to querying the vector database.
    Decides whether a database search is necessary, the optimized retrieval query,
    optional category or brand filters, and reasoning.
    """
    should_search: bool = True
    search_query: str = ""
    category_filter: Optional[str] = None
    brand_filter: Optional[str] = None
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossSellOffer:
    """
    Suggested complementary or bundle product with optional bundle discount.
    """
    target_product_id: str
    target_product_title: str
    suggested_product_id: str
    suggested_product_title: str
    reason: str
    bundle_discount_percent: float = 10.0  # e.g. 10% off for buying 2 products together


@dataclass
class SessionState:
    session_id: str
    customer_profile: CustomerProfile = field(default_factory=CustomerProfile)
    messages: List[ChatMessage] = field(default_factory=list)
    viewed_product_ids: List[str] = field(default_factory=list)
    interested_products: List[Dict[str, Any]] = field(default_factory=list)
    active_cross_sell: Optional[CrossSellOffer] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.messages.append(ChatMessage(role=role, content=content, metadata=metadata or {}))
        self.updated_at = datetime.utcnow().isoformat()


# Color and Cross-Sell Recommendation Matrix
COLOR_HARMONY_MAP = {
    "سفید": ["مشکی", "سفید", "طوسی", "آبی کاربنی", "صورتی"],
    "مشکی": ["مشکی", "سفید", "طوسی", "قرمز", "زرشکی"],
    "طوسی": ["مشکی", "سفید", "سرمه‌ای", "زرد"],
    "سرمه‌ای": ["سرمه‌ای", "سفید", "طوسی", "کرم"],
    "قهوه‌ای": ["کرم", "قهوه‌ای", "مشکی", "خاکی"],
    "آبی": ["سرمه‌ای", "سفید", "طوسی"],
    "قرمز": ["مشکی", "سفید"],
}

CATEGORY_COMPLEMENTARY_MAP = {
    "کفش": ["جوراب", "کفی کفش", "واکس و تمیزکننده"],
    "کتونی": ["جوراب اسپرت", "بند کفش", "اسپری بوگیر"],
    "شلوار": ["کمربند", "تی‌شرت", "پیراهن"],
    "پیراهن": ["کراوات", "کمربند", "شلوار"],
    "گوشی": ["قاب گوشی", "گلس و محافظ صفحه", "شارژر و کابل"],
    "لپ‌تاپ": ["ماوس", "کیف لپ‌تاپ", "پد ماوس"],
    "سیم و کابل": ["کلید و پریز", "چسب برق", "داکت"],
}


SHOPPING_ASSISTANT_SYSTEM_PROMPT = """شما «دستیار هوشمند و اختصاصی خرید ترب» (Torob Smart Shopping Assistant) هستید.
وظیفه شما راهنمایی حرفه‌ای، صمیمانه و متناسب با سلیقه مشتری برای انتخاب بهترین کالاها از میان محصولات موجود در پایگاه داده ترب است.

### 🔴 دستورالعمل حیاتی و الزامی اعراب‌گذاری (VOWELIZATION MANDATE FOR SPEECH):
پاسخ متنی شما مستقیماً توسط موتور صوتی مرورگر (Web Speech API) با صدای بلند برای کاربر خوانده می‌شود. موتورهای تبدیل متن به گفتار در زبان فارسی بدون اعراب کلمات را کاملاً غلط یا نامفهوم تلفظ می‌کنند.
به همین دلیل، **شما موظف هستید تمامی کلمات و جملات پاسخ خود را به طور کامل اعراب‌گذاری (حرکت‌گذاری با َ ، ِ ، ُ ، ً ، ٍ ، ٌ ، ّ ، ْ) کنید.**
- **هر کلمه باید دارای اعراب باشد**: حروف صدادار کوتاه (فتحه، کسره، ضمه) و سکون‌ها را به دقت بنویسید.
- **کسره اضافه را حتماً درج کنید**: برای اتصال صفت و موصوف یا مضاف و مضاف‌الیه حتماً کسره ِ بگذارید (مثال: «صَفْحِه‌یِ کِلاچِ لِیفانْ»، «قِیمَتِ آنْ»، «دِیسْکِ کِلاچِ»).
- **افعال و حروف ربط**: «می‌تَوانَد»، «شِناخْتِه می‌شَوَدْ»، «دارایِ»، «بَرایِ»، «کِه»، «بِه»، «اَزْ»، «دَرْ».
- **تنوین‌ها و تشدیدها**: «حَتْماً»، «لُطْفاً»، «مُعَرَّفِی»، «خُوشْ‌آمدِیدْ».

الگوی پاسخ شما باید عیناً مانند این نمونه‌ها باشد:
«بَلِه، صَفْحِه‌یِ کِلاچِ لِیفانْ دَرْ پایْگاهِ داده‌یِ تُرُبْ مَوْجُودْ اَسْتْ.
اِینْ مَحْصُولْ بِه نامِ دِیسْکْ وَ صَفْحِه‌یِ کِلاچِ لِیفانْ ۶۲۰ شِناخْتِه می‌شَوَدْ. قِیمَتِ آنْ حُدُودِ هَفْتْ مِلْیُونْ تُومانْ اَسْتْ.
اَگَرْ بِه اِطِّلاعاتِ بیشْتَرِی نِیازْ دارِیدْ، خُوشْحالْ می‌شَوَمْ کُمَکْ کُنَمْ!»

### اصول رفتاری و راهنمای مکالمه:
1. **صمیمیت و لحن مودبانه:** لحنی گرم، راهنما و در عین حال حرفه‌ای داشته باشید. به زبان فارسی روان صحبت کنید.
2. **استفاده از پروفایل بصری و دموگرافیک مشتری:**
   - اگر اطلاعاتی درباره ظاهر، رده سنی، جنسیت، استایل پوشش یا رنگ لباس‌های مشتری در کانتکست آمده است، پیشنهادهای خود را با سلیقه و سبک او هم‌راستا کنید.
   - هرگز مستقیماً نگویید «چون از عکست فهمیدم فلان سنی هستی!»؛ بلکه به شکلی طبیعی پیشنهادهایی سازگار با آن استایل (مثلاً اسپرت، رسمی یا رنگ‌های متناسب) ارائه دهید.
3. **مبتنی بر محصولات واقعی ترب (Grounding):**
   - فقط محصولاتی را معرفی و تحسین کنید که در بخش «محصولات یافت‌شده» (Retrieved Products) آمده‌اند یا موجودند. اطلاعات ساختگی ارائه ندهید.
   - برای هر محصول نام، قیمت تقریبی یا ویژگی‌های برجسته آن را ذکر کنید.
4. **استراتژی پیشنهاد مکمل (Cross-Sell / Add-on) و پیشنهاد تخفیف جفتی (Bundle Discount):**
   - اگر مشتری در حال خرید یا بررسی کالایی است (مثلاً کفش، گوشی، یا شلوار)، به شکلی محترمانه کالای مکمل متناسب با آن را با اعراب کامل پیشنهاد دهید:
     - برای مثال: پیشنهاد یک «جوراب با رنگ هماهنگ» برای کفش، یا «کابل/قاب» برای گوشی.
   - در صورت مناسب بودن، به مشتری آفر تخفیف خرید همزمان بدهید:
     «اگر این دو محصول را با هم تهیه کنید، کد تخفیف ویژه خرید جفتی (مثلاً ۱۰٪) برای شما اعمال می‌شود!»

### وضعیت پروفایل مشتری فعلی:
{customer_persona}

### محصولات مرتبط موجود در پایگاه داده:
{retrieved_products_context}

### پیشنهاد کالای مکمل فعال (Cross-Sell Opportunity):
{cross_sell_context}
"""

QUERY_FORMULATION_SYSTEM_PROMPT = """شما ماژول «تحلیل قصد و فرمولاسیون جستجوی پایگاه‌داده» برای موتور خرید ترب هستید.
وظیفه شما این است که پیش از مراجعه به پایگاه‌داده برداری/هیبریدی، پیام مشتری، تاریخچه گفتگو و پروفایل ظاهری مشتری را بررسی کنید و مشخص کنید:
1. آیا برای پاسخ به کاربر نیاز به جستجوی محصول در دیتابیس هست یا خیر؟ (مثلاً برای سلام، احوالپرسی، تشکر یا سوالات عمومی که ربطی به کالا ندارند، should_search = false است).
2. اگر بله، دقیق‌ترین عبارت جستجو (search_query) برای بازیابی محصولات در دیتابیس چیست؟
   - نیت کاربر را از عبارات محاوره‌ای به کلمات کلیدی مشخص کالا تبدیل کنید (مثلاً: «یه چیز شیک برای دویدن می‌خوام که به لباسم بیاد» -> با توجه به رنگ لباس مشکی و استایل اسپرت -> «کفش کتانی دویدن مشکی اسپرت»).
   - کلمات زائد و احوالپرسی‌ها را از کوئری حذف کنید.
3. فیلتر category_filter را فقط و فقط زمانی مقداردهی کنید که کاربر صریحاً به یک دسته‌بندی خاص اشاره کرده باشد و از نام دقیق آن مطمئن هستید. در غیر این صورت و در حالت‌های کلی یا حدسی، حتماً مقدار category_filter را null قرار دهید تا جستجوی متنی و برداری محدود نشود.
4. برند درخواستی (در صورت ذکر شدن) را در brand_filter استخراج کنید؛ در غیر این صورت null بگذارید.

خروجی شما باید منحصراً یک ساختار JSON معتبر با کلیدهای زیر باشد (بدون هیچ توضیح اضافه یا markdown غیراز JSON):
{{
    "should_search": true,
    "search_query": "عبارت بهینه‌شده جستجو برای دیتابیس",
    "category_filter": "نام دسته‌بندی یا null",
    "brand_filter": "نام برند یا null",
    "reasoning": "دلیل و توضیح کوتاه انتخاب این عبارت"
}}
"""


class ShoppingAgent:
    """
    Intelligent shopping assistant orchestrating RAG retrieval,
    customer vision profile context, cross-sell recommendation, and session memory.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        repository: Optional[Repository] = None,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.retriever = retriever
        self.repository = repository
        self.model_name = model_name
        self.api_key = api_key or os.getenv("BOXAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("BOXAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://ai.boxapi.ir/ai/v1"
        ).rstrip("/")

        # In-memory session store (can be replaced with Redis or DB in production)
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        customer_profile: Optional[CustomerProfile] = None,
    ) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        if sid not in self.sessions:
            self.sessions[sid] = SessionState(
                session_id=sid,
                customer_profile=customer_profile or CustomerProfile(),
            )
        elif customer_profile:
            # Update customer profile if new camera data arrived
            self.sessions[sid].customer_profile = customer_profile
        return self.sessions[sid]

    def find_cross_sell_opportunity(
        self,
        top_product: Document,
        customer_profile: CustomerProfile,
    ) -> Optional[CrossSellOffer]:
        """
        Calculates complementary cross-sell items based on color harmony and category mappings.
        """
        title = top_product.metadata.get("persian_title", "")
        category = top_product.metadata.get("category_name", "")
        pid = top_product.metadata.get("product_id", "")
        content = top_product.page_content

        target_cat = None
        complementary_keywords = []
        for cat_key, comps in CATEGORY_COMPLEMENTARY_MAP.items():
            if cat_key in title or cat_key in category or cat_key in content:
                target_cat = cat_key
                complementary_keywords = comps
                break

        if not complementary_keywords:
            return None

        # Determine preferred color harmony
        preferred_color = "مشکی"
        for col in COLOR_HARMONY_MAP.keys():
            if col in title or col in content:
                matching_colors = COLOR_HARMONY_MAP.get(col, [])
                if matching_colors:
                    preferred_color = matching_colors[0]
                break

        # Query complementary product via retriever
        cross_query = f"{complementary_keywords[0]} رنگ {preferred_color}"
        comp_docs = self.retriever.retrieve(query=cross_query, limit=2, group_by_product=True)
        if not comp_docs:
            return None

        suggested = comp_docs[0]
        s_title = suggested.metadata.get("persian_title", cross_query)
        s_id = suggested.metadata.get("product_id", "")

        return CrossSellOffer(
            target_product_id=pid,
            target_product_title=title,
            suggested_product_id=s_id,
            suggested_product_title=s_title,
            reason=f"هماهنگی رنگ ({preferred_color}) و تکمیل خرید برای دسته‌بندی {target_cat}",
            bundle_discount_percent=10.0,
        )

    def _format_products_context(self, docs: List[Document]) -> str:
        if not docs:
            return "هیچ کالای منطبقی در این جستجو یافت نشد."

        lines = []
        for i, doc in enumerate(docs, start=1):
            m = doc.metadata
            title = m.get("persian_title", "نامشخص")
            pid = m.get("product_id", "")
            cat = m.get("category_name", "")
            brand = m.get("brand_name", "")

            # Check price/offers from repository if available
            price_info = ""
            if self.repository and pid:
                cheapest = self.repository.offers.get_cheapest_for_product(pid)
                if cheapest and cheapest.price_toman > 0:
                    price_info = f" | ارزان‌ترین قیمت: {cheapest.price_toman:,} تومان"

            score = m.get("rerank_score", m.get("score", 0.0))
            lines.append(
                f"{i}. [شناسه: {pid}] {title} (برند: {brand or 'متفرقه'} | دسته: {cat}{price_info} | امتیاز تطابق: {score})\n"
                f"   مشخصات: {doc.page_content.strip()[:300]}..."
            )
        return "\n\n".join(lines)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Calls OpenAI/BoxAI compatible chat completion API.
        Falls back to local rule-based response generation if offline or API key absent.
        """
        if not self.api_key:
            return self._local_fallback_response(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 1200,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"LLM API request to {self.base_url}/chat/completions failed: {e}. Falling back to rule-based response.")
            return self._local_fallback_response(messages)

    def _local_fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Fallback response when LLM API is unavailable."""
        system_content = messages[0]["content"] if messages else ""
        user_query = messages[-1]["content"] if messages else ""

        # Extract product suggestions directly from context
        return (
            f"سلام و احترام! در خصوص درخواست شما («{user_query}»)، بر اساس محصولات موجود در ترب این گزینه‌ها پیشنهاد می‌شوند:\n\n"
            f"کالاهای متناسب برای شما پیدا شدند و می‌توانید جزئیات مشخصات و قیمت‌های آنها را در بخش بالا مشاهده بفرمایید. "
            f"اگر به دنبال رنگ یا ویژگی خاصی هستید، بفرمایید تا دقیق‌تر راهنماییتان کنم!"
        )

    def formulate_search_intent(
        self,
        user_message: str,
        session: SessionState,
    ) -> SearchIntent:
        """
        Ask the agent to determine what to look for in the database before querying the retriever:
        1. Classifies if user message is a conversational greeting/chitchat or product-seeking intent.
        2. Formulates an optimized keyword/semantic database query incorporating customer context.
        3. Identifies candidate category or brand filters if applicable.
        """
        user_text = user_message.strip()

        # Quick rule check for pure conversational greetings
        greetings = {"سلام", "درود", "سلام وقت بخیر", "سلام خسته نباشید", "مرسی", "ممنون", "تشکر", "خداحافظ"}
        if user_text.lower() in greetings or (len(user_text.split()) <= 2 and any(g in user_text for g in greetings) and not any(p in user_text for p in ["خرید", "قیمت", "کفش", "گوشی", "لپ‌تاپ", "سیم", "می‌خوام", "دارید"])):
            return SearchIntent(
                should_search=False,
                search_query="",
                reasoning="پیام کاربر صرفاً خوش‌آمدگویی یا احوالپرسی اولیه است و نیازی به واکشی دیتابیس ندارد.",
            )

        # Assemble prompt for LLM query formulation
        persona_text = session.customer_profile.to_persona_summary()
        recent_history = [
            f"{m.role}: {m.content}"
            for m in session.messages[-4:]
            if m.content != user_message
        ]
        history_str = "\n".join(recent_history) if recent_history else "شروع گفتگو"

        prompt = (
            f"پروفایل ظاهری مشتری:\n{persona_text}\n\n"
            f"تاریخچه اخیر گفتگو:\n{history_str}\n\n"
            f"پیام جدید مشتری:\n{user_message}\n\n"
            f"لطفاً به عنوان ماژول تحلیل قصد، خروجی JSON مشخص‌شده را برای استخراج کوئری پایگاه داده ارائه دهید."
        )

        messages = [
            {"role": "system", "content": QUERY_FORMULATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if self.api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 250,
                }
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    # Strip any markdown code blocks if wrapped
                    if content.startswith("```"):
                        lines = content.splitlines()
                        content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
                    data = json.loads(content)
                    return SearchIntent(
                        should_search=bool(data.get("should_search", True)),
                        search_query=str(data.get("search_query", user_message)).strip(),
                        category_filter=data.get("category_filter") or None,
                        brand_filter=data.get("brand_filter") or None,
                        reasoning=str(data.get("reasoning", "")),
                    )
            except Exception:
                pass

        # Intelligent local fallback for query formulation
        return self._local_formulate_search_intent(user_message, session)

    def _local_formulate_search_intent(
        self,
        user_message: str,
        session: SessionState,
    ) -> SearchIntent:
        """
        Rule-based local fallback to formulate database search query from user message and customer profile.
        """
        query_words = user_message.strip()

        # Strip conversational prefix phrases
        prefixes_to_strip = [
            "سلام", "درود", "لطفا", "لطفاً", "بی زحمت", "بی‌زحمت",
            "می‌خواستم ببینم", "می‌خوام بدونم", "دنبال", "یه دونه", "یک عدد",
            "می‌خوام", "میخوام", "برای من", "پیشنهاد بدید", "هستید", "دارید",
        ]
        cleaned_query = query_words
        for p in prefixes_to_strip:
            cleaned_query = cleaned_query.replace(p, " ")

        cleaned_query = " ".join(cleaned_query.split())

        # If user is asking for items matching their clothing / style:
        # e.g., "کفش مناسب با لباسم"
        profile = session.customer_profile
        extra_terms = []

        if any(w in user_message for w in ["لباسم", "ست", "هماهنگ", "پوششم"]):
            if profile.clothing_colors_detected:
                # Add primary detected clothing color
                extra_terms.append(profile.clothing_colors_detected[0])
            if profile.style_preference and profile.style_preference not in cleaned_query:
                extra_terms.append(profile.style_preference)

        if extra_terms and cleaned_query:
            formulated_query = f"{cleaned_query} {' '.join(extra_terms)}".strip()
        else:
            formulated_query = cleaned_query or user_message.strip()

        # Detect category filter
        category_filter = None
        for cat in CATEGORY_COMPLEMENTARY_MAP.keys():
            if cat in user_message:
                category_filter = cat
                break

        return SearchIntent(
            should_search=True,
            search_query=formulated_query,
            category_filter=category_filter,
            brand_filter=None,
            reasoning="استخراج محلی کلمات کلیدی کالا و تطبیق با استایل و رنگ پروفایل کاربر",
        )

    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        customer_profile: Optional[CustomerProfile] = None,
        retrieval_limit: int = 4,
    ) -> Dict[str, Any]:
        """
        Full conversational turn:
        1. Access or create session with customer profile.
        2. First ask agent/formulator what to look for in the database (SearchIntent).
        3. If should_search is True, perform hybrid retrieval + rerank with the formulated query.
        4. Detect cross-sell & bundle opportunities.
        5. Construct rich contextual prompt and call LLM.
        6. Save session state and return structured response.
        """
        session = self.get_or_create_session(session_id=session_id, customer_profile=customer_profile)
        session.add_message(role="user", content=user_message)

        # 1. Ask agent what to look for in database (Query formulation & intent classification)
        search_intent = self.formulate_search_intent(user_message=user_message, session=session)

        retrieved_docs: List[Document] = []
        if search_intent.should_search and search_intent.search_query:
            # Prepare optional metadata filters identified by the agent
            filter_dict = {}
            if search_intent.category_filter:
                filter_dict["category_name"] = search_intent.category_filter
            if search_intent.brand_filter:
                filter_dict["brand_name"] = search_intent.brand_filter

            # 2. Hybrid Retrieval from Qdrant using formulated database query
            retrieved_docs = self.retriever.retrieve(
                query=search_intent.search_query,
                limit=retrieval_limit,
                filter_dict=filter_dict if filter_dict else None,
                group_by_product=True,
                use_reranker=True,
            )

            # Fallback relaxation: If strict category filter returned 0 results, retry without category filter
            # (LLMs often guess categories like 'مبلمان' that might differ from exact category_name in catalog)
            if not retrieved_docs and filter_dict.get("category_name"):
                relaxed_filters = {k: v for k, v in filter_dict.items() if k != "category_name"}
                retrieved_docs = self.retriever.retrieve(
                    query=search_intent.search_query,
                    limit=retrieval_limit,
                    filter_dict=relaxed_filters if relaxed_filters else None,
                    group_by_product=True,
                    use_reranker=True,
                )

            # Track product IDs in session
            for doc in retrieved_docs:
                pid = doc.metadata.get("product_id")
                if pid and pid not in session.viewed_product_ids:
                    session.viewed_product_ids.append(pid)

        # 3. Check Cross-Sell & Bundle Discount Opportunity
        cross_sell: Optional[CrossSellOffer] = None
        if retrieved_docs:
            cross_sell = self.find_cross_sell_opportunity(retrieved_docs[0], session.customer_profile)
            session.active_cross_sell = cross_sell

        cross_sell_text = "در این مرحله پیشنهاد مکمل خاصی تعریف نشده است."
        if cross_sell:
            cross_sell_text = (
                f"پیشنهاد کالای مکمل فعال است:\n"
                f"- کالای اصلی: {cross_sell.target_product_title}\n"
                f"- کالای مکمل پیشنهادی: {cross_sell.suggested_product_title}\n"
                f"- دلیل تطابق: {cross_sell.reason}\n"
                f"- آفر تخفیف: {cross_sell.bundle_discount_percent}٪ تخفیف روی خرید همزمان هر دو کالا.\n"
                f"لطفاً در انتهای پاسختان این پیشنهاد را با لحنی جذاب و دوستانه به مشتری معرفی کنید."
            )

        # 4. Build System Prompt
        products_context = (
            self._format_products_context(retrieved_docs)
            if retrieved_docs
            else "هیچ محصولی در این نوبت استعلام یا بازیابی نشد (یا پیام غیرمرتبط با کالای خاص بوده است)."
        )
        system_prompt = SHOPPING_ASSISTANT_SYSTEM_PROMPT.format(
            customer_persona=session.customer_profile.to_persona_summary(),
            retrieved_products_context=products_context,
            cross_sell_context=cross_sell_text,
        )

        # 5. Assemble LLM Conversation History (rolling window)
        llm_messages = [{"role": "system", "content": system_prompt}]
        for msg in session.messages[-6:]:
            llm_messages.append({"role": msg.role, "content": msg.content})

        # Strict end-of-context directive to guarantee full vowelization regardless of older history
        llm_messages.append({
            "role": "system",
            "content": (
                "دستور اکید برای این پاسخ: حتماً تمامی کلمات، افعال و جملات پاسخ را به صورت کامل با حرکات و اعراب (َ ِ ُ ً ٍ ٌ ّ ْ) بنویسید تا موتور گفتاری بدون غلط تلفظ کند. الگو: «بَلِه، صَفْحِه‌یِ کِلاچِ لِیفانْ مَوْجُودْ اَسْتْ...»."
            ),
        })

        # 6. Execute LLM Call
        assistant_reply = self._call_llm(llm_messages)
        session.add_message(
            role="assistant",
            content=assistant_reply,
            metadata={
                "search_intent": search_intent.to_dict(),
                "retrieved_count": len(retrieved_docs),
                "has_cross_sell": cross_sell is not None,
            },
        )

        # Enrich retrieved products with real shop details from repository
        retrieved_products_data = []
        for d in retrieved_docs:
            meta = d.metadata
            pid = meta.get("product_id", "")
            title = meta.get("persian_title") or "محصول ترب"
            score = meta.get("rerank_score", meta.get("score", 0.0))

            item_data = {
                "product_id": pid,
                "title": title,
                "score": score,
                "category_name": meta.get("category_name"),
                "brand_name": meta.get("brand_name"),
            }

            if self.repository and pid:
                prod_details = (
                    self.repository.get_product_details_dict(pid)
                    if hasattr(self.repository, "get_product_details_dict")
                    else None
                )
                if prod_details:
                    item_data["min_price"] = prod_details.get("min_price")
                    item_data["min_price_formatted"] = prod_details.get("min_price_formatted")
                    item_data["offers_count"] = prod_details.get("offers_count")
                    item_data["image_url"] = prod_details.get("primary_image")
                    item_data["shops"] = prod_details.get("shops", [])
                    item_data["specs"] = prod_details.get("specs", [])
                else:
                    cheapest = self.repository.offers.get_cheapest_for_product(pid)
                    if cheapest and cheapest.price_toman > 0:
                        item_data["min_price"] = cheapest.price_toman
                        item_data["min_price_formatted"] = f"از {cheapest.price_toman:,} تومان"
                    offers = self.repository.offers.get_by_product_id(pid)
                    shops_data = []
                    for off in offers:
                        sh = self.repository.shops.get_by_id(off.shop_id)
                        if sh:
                            shops_data.append({
                                "shop_id": sh.shop_id,
                                "shop_name": sh.name,
                                "price": off.price_toman,
                                "price_formatted": f"{off.price_toman:,} تومان" if off.price_toman > 0 else "استعلام",
                                "city": sh.location_label or "تهران",
                                "warranty": off.warranty or "گارانتی معتبر فروشگاه",
                                "supports_fast_delivery": bool(sh.supports_fast_delivery),
                            })
                    item_data["shops"] = shops_data
                    item_data["offers_count"] = len(shops_data)

            retrieved_products_data.append(item_data)

        return {
            "session_id": session.session_id,
            "reply": assistant_reply,
            "search_intent": search_intent.to_dict(),
            "retrieved_products": retrieved_products_data,
            "cross_sell_offer": asdict(cross_sell) if cross_sell else None,
            "customer_profile": session.customer_profile.to_dict(),
            "history_length": len(session.messages),
        }
