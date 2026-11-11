"""Безсетевой тест подписок и разбора T-Invest market-data stream."""
import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import tinkoff_client


class StreamTypesTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscribes_to_prices_and_order_book_and_parses_uid(self):
        captured = []

        class Payload:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class SubscriptionAction:
            SUBSCRIPTION_ACTION_SUBSCRIBE = "SUBSCRIBE"

        sdk = SimpleNamespace(
            MarketDataRequest=Payload,
            SubscribeOrderBookRequest=Payload,
            SubscribeLastPriceRequest=Payload,
            OrderBookInstrument=Payload,
            LastPriceInstrument=Payload,
            SubscriptionAction=SubscriptionAction,
        )

        last_event = SimpleNamespace(
            last_price=SimpleNamespace(
                instrument_uid="uid-test", figi="", price="100.50"),
            orderbook=None,
        )
        book_event = SimpleNamespace(
            last_price=None,
            orderbook=SimpleNamespace(
                instrument_uid="uid-test", figi="",
                bids=[SimpleNamespace(price="100.40", quantity=40)],
                asks=[SimpleNamespace(price="100.60", quantity=20)],
                is_consistent=True,
            ),
        )

        class MarketDataStreamService:
            def market_data_stream(self, request_iterator):
                async def response_stream():
                    # Проверяем, что обе подписки отправляются отдельными
                    # MarketDataRequest, а client-side поток остаётся живым.
                    first = await request_iterator.__anext__()
                    second = await request_iterator.__anext__()
                    captured.extend([first, second])
                    yield last_event
                    yield book_event
                    while True:
                        await asyncio.sleep(1)
                return response_stream()

        client = tinkoff_client.TinkoffClient(
            SimpleNamespace(rate_limit_rpm=90), SimpleNamespace())
        client._api = SimpleNamespace(market_data_stream=MarketDataStreamService())
        client._universe["TEST"] = {
            "figi": "figi-test", "uid": "uid-test", "lot": 1,
            "name": "Test instrument", "class_code": "TQBR",
        }

        with patch.object(tinkoff_client.importlib, "import_module", return_value=sdk), \
             patch.object(tinkoff_client, "quotation_to_decimal",
                          side_effect=lambda value: Decimal(str(value))):
            started = await client.start_scalp_feed(["TEST"])
            self.assertTrue(started)
            await asyncio.sleep(0.02)

            self.assertEqual(len(captured), 2)
            self.assertIsNotNone(captured[0].subscribe_order_book_request)
            self.assertIsNotNone(captured[1].subscribe_last_price_request)
            book_request = captured[0].subscribe_order_book_request
            self.assertEqual(book_request.instruments[0].instrument_id, "uid-test")
            self.assertEqual(book_request.instruments[0].depth, 5)

            quote = client.get_scalp_quote("TEST")
            self.assertIsNotNone(quote)
            self.assertAlmostEqual(quote["price"], 100.50)
            self.assertAlmostEqual(quote["bid"], 100.40)
            self.assertAlmostEqual(quote["ask"], 100.60)
            self.assertGreaterEqual(quote["price_age_sec"], 0)
            self.assertGreaterEqual(quote["book_age_sec"], 0)
            await client.stop_scalp_feed()
            self.assertFalse(client.scalp_feed_ready)


if __name__ == "__main__":
    unittest.main()
