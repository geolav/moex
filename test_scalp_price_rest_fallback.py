"""Проверки пакетного REST-fallback last-price для скальп-режима."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tinkoff_client import TinkoffClient


class ScalpRestPriceTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        client = object.__new__(TinkoffClient)
        client._api = SimpleNamespace(market_data=SimpleNamespace())
        client._instruments = {
            "SBER": {"uid": "uid-sber", "figi": "figi-sber"},
            "LKOH": {"uid": "uid-lkoh", "figi": "figi-lkoh"},
        }
        client._ticker_by_figi = {"figi-sber": "SBER", "figi-lkoh": "LKOH"}
        client._ticker_by_uid = {"uid-sber": "SBER", "uid-lkoh": "LKOH"}
        client._scalp_quotes = {
            "SBER": {"ticks": [], "price": 300.0},
            "LKOH": {"ticks": [], "price": 7000.0},
        }
        client._scalp_price_rest_last = 0.0
        client._scalp_price_rest_lock = asyncio.Lock()
        client._stream_number = lambda value: float(value)
        client._call = AsyncMock(return_value=SimpleNamespace(last_prices=[
            SimpleNamespace(figi="figi-sber", instrument_uid="uid-sber", price=301.0),
            SimpleNamespace(figi="figi-lkoh", instrument_uid="uid-lkoh", price=7010.0),
        ]))
        return client

    async def test_rest_batch_fills_stale_prices_and_tick_buffers(self):
        client = self.client()
        count = await client.refresh_stale_prices(["SBER", "LKOH"], refresh_sec=2, max_age_sec=2)
        self.assertEqual(count, 2)
        self.assertEqual(client._call.await_count, 1)
        self.assertEqual(client._call.await_args.args[0], "get_last_prices(scalp_rest_batch)")
        for ticker, expected in (("SBER", 301.0), ("LKOH", 7010.0)):
            quote = client.get_scalp_quote(ticker)
            self.assertEqual(quote["price"], expected)
            self.assertEqual(quote["price_source"], "rest")
            self.assertEqual(quote["price_seq"], 1)
            self.assertEqual(len(quote["ticks"]), 1)
            self.assertLess(quote["price_age_sec"], 1.0)

    async def test_fresh_stream_price_is_not_overwritten(self):
        client = self.client()
        client._scalp_quotes["SBER"].update({
            "price": 302.0, "price_source": "stream",
            "price_updated_mono": asyncio.get_running_loop().time(),
            "price_seq": 7,
        })
        count = await client.refresh_stale_prices(["SBER", "LKOH"], refresh_sec=2, max_age_sec=5)
        self.assertEqual(count, 1)
        self.assertEqual(client._scalp_quotes["SBER"]["price"], 302.0)
        self.assertEqual(client._scalp_quotes["SBER"]["price_source"], "stream")
        self.assertEqual(client._scalp_quotes["LKOH"]["price"], 7010.0)

    async def test_global_throttle_avoids_api_spam(self):
        client = self.client()
        await client.refresh_stale_prices(["SBER", "LKOH"], refresh_sec=10)
        client._call.reset_mock()
        count = await client.refresh_stale_prices(["SBER", "LKOH"], refresh_sec=10)
        self.assertEqual(count, 0)
        self.assertEqual(client._call.await_count, 0)
