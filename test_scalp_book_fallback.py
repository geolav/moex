"""Регрессия v1.17.2: видимость отказа подписки на стакан и REST-фолбэк.

Сценарий владельца: сервер принимал last_price, но стакан по стриму
не приходил и НИКАК не логировался (ответы на подписки игнорировались),
а детектор браковал все микродна с "стакан устарел".
"""
from __future__ import annotations

import asyncio
import logging
import time
import unittest
from types import SimpleNamespace

from tinkoff_client import TinkoffClient


def make_client(scalp_cfg: dict | None = None) -> TinkoffClient:
    cfg = SimpleNamespace(
        rate_limit_rpm=600,
        http_timeout_sec=5,
        scalp_book_rest_refresh_sec=20,
        scalp_rest_book_max_age_sec=45,
    )
    cfg.__dict__.update(scalp_cfg or {})
    state = SimpleNamespace(use_sandbox=True)
    client = TinkoffClient(cfg, state)
    client._instruments["SBER"] = {"figi": "figi-sber", "uid": "uid-sber",
                                   "lot": 10, "name": "Сбербанк"}
    client._ticker_by_figi["figi-sber"] = "SBER"
    client._ticker_by_uid["uid-sber"] = "SBER"
    return client


def fake_book_response(bid: float = 100.00, ask: float = 100.02):
    return SimpleNamespace(
        bids=[SimpleNamespace(price=bid, quantity=1200)],
        asks=[SimpleNamespace(price=ask, quantity=700)],
    )


class SubscriptionStatusLoggingTests(unittest.TestCase):
    def test_rejected_orderbook_subscription_is_logged(self):
        client = make_client()
        event = SimpleNamespace(
            subscribe_last_price_response=None,
            subscribe_order_book_response=SimpleNamespace(
                order_book_subscriptions=[
                    SimpleNamespace(
                        instrument_uid="uid-sber", figi="figi-sber",
                        subscription_status=SimpleNamespace(
                            name="SUBSCRIPTION_STATUS_LIMIT_IS_EXCEEDED"),
                    ),
                ]),
            last_price=None,
            orderbook=None,
        )
        with self.assertLogs("tinkoff", level="WARNING") as captured:
            client._handle_scalp_stream_event(event)
        text = "\n".join(captured.output)
        self.assertIn("ОТКЛОНИЛ", text)
        self.assertIn("SBER", text)
        self.assertIn("LIMIT_IS_EXCEEDED", text)

    def test_successful_subscription_is_logged_once(self):
        client = make_client()
        ok = SimpleNamespace(name="SUBSCRIPTION_STATUS_SUCCESS")
        event = SimpleNamespace(
            subscribe_last_price_response=SimpleNamespace(
                last_price_subscriptions=[
                    SimpleNamespace(instrument_uid="uid-sber",
                                    figi="figi-sber", subscription_status=ok)]),
            subscribe_order_book_response=SimpleNamespace(
                order_book_subscriptions=[
                    SimpleNamespace(instrument_uid="uid-sber",
                                    figi="figi-sber", subscription_status=ok)]),
            last_price=None,
            orderbook=None,
        )
        with self.assertLogs("tinkoff", level="INFO") as captured:
            client._handle_scalp_stream_event(event)
            # Повторные события не должны логировать подписки заново.
            client._handle_scalp_stream_event(event)
        info = [line for line in captured.output
                if "подтверждена" in line]
        self.assertEqual(len(info), 2)  # last_price + orderbook, один раз


class RestBookFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_stale_books_fills_quote(self):
        client = make_client()
        calls: list[str] = []

        class FakeMarketData:
            async def get_order_book(self, figi=None, depth=5, instrument_id=None):
                calls.append(figi or instrument_id)
                return fake_book_response()

        client._api = SimpleNamespace(market_data=FakeMarketData())
        # Стримовая цена есть, стакана нет вообще.
        client._scalp_quotes["SBER"] = {
            "ticks": __import__("collections").deque(maxlen=120),
            "price": 100.01, "price_updated_mono": time.monotonic(),
        }
        updated = await client.refresh_stale_books(["SBER"])
        self.assertEqual(updated, 1)
        self.assertEqual(calls, ["figi-sber"])
        quote = client.get_scalp_quote("SBER")
        self.assertEqual(quote["book_source"], "rest")
        self.assertLessEqual(quote["book_age_sec"], 1.0)
        self.assertAlmostEqual(quote["bid"], 100.00)
        self.assertAlmostEqual(quote["ask"], 100.02)
        spread = client.get_spread_pct("SBER")
        self.assertIsNotNone(spread)
        self.assertGreater(spread, 0.0)

    async def test_fresh_stream_book_is_not_overwritten(self):
        client = make_client()

        class FailMarketData:
            async def get_order_book(self, **kwargs):
                raise AssertionError("REST не должен вызываться при живом стакане")

        client._api = SimpleNamespace(market_data=FailMarketData())
        now = time.monotonic()
        client._scalp_quotes["SBER"] = {
            "price": 100.01, "price_updated_mono": now,
            "book_updated_mono": now, "book_source": "stream",
            "bid": 100.0, "ask": 100.02, "bid_volume": 10, "ask_volume": 10,
            "book_consistent": True, "spread_pct": 0.02,
            "ticks": __import__("collections").deque(maxlen=120),
        }
        updated = await client.refresh_stale_books(["SBER"])
        self.assertEqual(updated, 0)
        self.assertEqual(client.get_scalp_quote("SBER")["book_source"], "stream")

    async def test_rest_failure_is_silent_and_throttled(self):
        client = make_client()

        class BrokenMarketData:
            async def get_order_book(self, **kwargs):
                raise RuntimeError("нет сети")

        client._api = SimpleNamespace(market_data=BrokenMarketData())
        client._scalp_quotes["SBER"] = {"price": 100.01,
                                        "price_updated_mono": time.monotonic()}
        updated = await client.refresh_stale_books(["SBER"])
        self.assertEqual(updated, 0)
        # Повторный вызов сразу же - не должен повторять запрос (тротлинг).
        updated = await client.refresh_stale_books(["SBER"])
        self.assertEqual(updated, 0)

    async def test_spread_fallback_chain_rest_then_iss(self):
        client = make_client()
        # Нет ни стримового, ни REST-стакана, но есть свежий ISS-снимок.
        client._spread_cache["SBER"] = (0.06, time.monotonic())
        spread = client.get_spread_pct("SBER")
        self.assertAlmostEqual(spread, 0.06)
        # Устаревший ISS-снимок - None.
        client._spread_cache["SBER"] = (0.06, time.monotonic() - 600)
        self.assertIsNone(client.get_spread_pct("SBER"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
