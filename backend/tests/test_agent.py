import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from rag.agent import ShoppingAgent, CustomerProfile, CrossSellOffer, SessionState


class TestShoppingAgent(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_repo = MagicMock()
        self.agent = ShoppingAgent(
            retriever=self.mock_retriever,
            repository=self.mock_repo,
        )

    def test_session_creation_and_memory(self):
        profile = CustomerProfile(
            gender="مردانه",
            age_group="جوان (۲۰-۳۰)",
            style_preference="اسپرت",
            clothing_colors_detected=["مشکی", "سفید"],
        )
        session = self.agent.get_or_create_session(session_id="test_sess_1", customer_profile=profile)
        self.assertEqual(session.session_id, "test_sess_1")
        self.assertEqual(session.customer_profile.style_preference, "اسپرت")

        session.add_message("user", "سلام دنبال کفش ورزشی هستم")
        session.add_message("assistant", "سلام! چه سایزی مد نظرتونه؟")
        self.assertEqual(len(session.messages), 2)

    def test_cross_sell_socks_for_shoes(self):
        shoe_doc = Document(
            page_content="کفش کتانی پیاده‌روی مشکی نایک کیفیت بالا",
            metadata={
                "product_id": "shoe_123",
                "persian_title": "کفش ورزشی مشکی نایک",
                "category_name": "کفش ورزشی",
            },
            id="shoe_123",
        )

        sock_doc = Document(
            page_content="جوراب ساق بلند اسپرت مشکی نخی",
            metadata={
                "product_id": "sock_456",
                "persian_title": "جوراب اسپرت مشکی",
                "category_name": "جوراب",
            },
            id="sock_456",
        )

        # When searching for socks, return sock_doc
        self.mock_retriever.retrieve.return_value = [sock_doc]

        profile = CustomerProfile(style_preference="اسپرت")
        offer = self.agent.find_cross_sell_opportunity(top_product=shoe_doc, customer_profile=profile)

        self.assertIsNotNone(offer)
        self.assertEqual(offer.target_product_id, "shoe_123")
        self.assertEqual(offer.suggested_product_id, "sock_456")
        self.assertIn("هماهنگی رنگ", offer.reason)
        self.assertEqual(offer.bundle_discount_percent, 10.0)

    def test_chat_turn_integration(self):
        shoe_doc = Document(
            page_content="کفش چرم مردانه قهوه‌ای رسمی",
            metadata={
                "product_id": "p_shoe",
                "persian_title": "کفش چرم طبیعی قهوه‌ای",
                "category_name": "کفش",
                "score": 0.95,
            },
            id="p_shoe",
        )
        self.mock_retriever.retrieve.return_value = [shoe_doc]

        response = self.agent.chat(
            user_message="یک کفش چرم مجلسی می‌خوام",
            session_id="sess_100",
            customer_profile=CustomerProfile(gender="مرد", style_preference="رسمی"),
        )

        self.assertEqual(response["session_id"], "sess_100")
        self.assertIn("reply", response)
        self.assertIn("search_intent", response)
        self.assertTrue(response["search_intent"]["should_search"])
        self.assertTrue(len(response["search_intent"]["search_query"]) > 0)
        self.assertEqual(len(response["retrieved_products"]), 1)
        self.assertEqual(response["retrieved_products"][0]["product_id"], "p_shoe")
        self.assertEqual(response["history_length"], 2)  # 1 user + 1 assistant

    def test_pure_greeting_skips_database_retrieval(self):
        # A simple greeting should not trigger database search
        response = self.agent.chat(
            user_message="سلام",
            session_id="sess_greeting",
        )
        self.assertFalse(response["search_intent"]["should_search"])
        self.assertEqual(len(response["retrieved_products"]), 0)
        self.mock_retriever.retrieve.assert_not_called()

    def test_query_formulation_incorporates_customer_profile(self):
        profile = CustomerProfile(
            style_preference="اسپرت",
            clothing_colors_detected=["مشکی"],
        )
        session = self.agent.get_or_create_session(session_id="sess_style", customer_profile=profile)
        intent = self.agent.formulate_search_intent(
            user_message="یه کفش ست با لباسم می‌خوام",
            session=session,
        )
        self.assertTrue(intent.should_search)
        self.assertIn("مشکی", intent.search_query)
        self.assertIn("کفش", intent.search_query)

    def test_format_products_context_with_offer_price(self):
        from repository.offers import Offer
        mock_offer = Offer(
            offer_id="off_1",
            product_id="p_shoe",
            shop_id="s_1",
            price_toman=500000,
        )
        self.mock_repo.offers.get_cheapest_for_product.return_value = mock_offer
        doc = Document(
            page_content="کفش چرم",
            metadata={"product_id": "p_shoe", "persian_title": "کفش چرم"},
        )
        context = self.agent._format_products_context([doc])
        self.assertIn("500,000 تومان", context)
        self.assertEqual(mock_offer.price, 500000)


if __name__ == "__main__":
    unittest.main()
