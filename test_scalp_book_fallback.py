"""Проверки REST-fallback стакана в восстановленной сборке."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tinkoff_client import TinkoffClient


class ScalpRestBookTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        client = object.__new__(TinkoffClient)
        client._api = SimpleNamespace(market_data=SimpleNamespace())
        client._instruments = {"SBER": {"uid": "uid-sber", "figi": "figi-sber"}}
        client._scalp_quotes = {"SBER": {"ticks": [], "price": 300.0}}
        client._scalp_book_rest_last = {}
        client._scalp_book_rest_cursor = 0
        client._scalp_book_rest_lock = asyncio.Lock()
        client._stream_number = lambda value: float(value)
        client._call = AsyncMock(return_value=SimpleNamespace(
            bids=[SimpleNamespace(price=299.9, quantity=100),
                  SimpleNamespace(price=299.8, quantity=50)],
            asks=[SimpleNamespace(price=300.1, quantity=80),
                  SimpleNamespace(price=300.2, quantity=20)],
        ))
        return client

    async def test_rest_fallback_populates_book_and_source(self):
        client = self.client()
        count = await client.refresh_stale_books(["SBER"], max_requests=1)
        self.assertEqual(count, 1)
        quote = client.get_scalp_quote("SBER")
        self.assertEqual(quote["book_source"], "rest")
        self.assertAlmostEqual(quote["bid"], 299.9)
        self.assertAlmostEqual(quote["ask"], 300.1)
        self.assertGreater(quote["spread_pct"], 0)

    async def test_live_stream_book_is_not_overwritten(self):
        client = self.client()
        client._scalp_quotes["SBER"].update({
            "bid": 299.95, "ask": 300.05, "bid_volume": 100,
            "ask_volume": 90, "book_source": "stream",
            "book_updated_mono": asyncio.get_running_loop().time(),
            "book_consistent": True,
        })
        count = await client.refresh_stale_books(["SBER"], max_requests=1)
        self.assertEqual(count, 0)
        self.assertEqual(client._call.await_count, 0)
        self.assertEqual(client._scalp_quotes["SBER"]["book_source"], "stream")


if __name__ == "__main__":
    unittest.main()
