"""Backend switch must avoid constructing or calling Groq in local mode."""
import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys

# The local backend must be testable without installing/importing the Groq SDK.
# If the SDK is absent, provide import-only exception/client stubs; constructing
# the client is a test failure because local mode must never do so.
try:
    import groq  # noqa: F401
except ModuleNotFoundError:
    class _GroqError(Exception):
        pass

    class _UnexpectedGroqClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Groq client must not be constructed in local mode")

    sys.modules["groq"] = SimpleNamespace(
        APIConnectionError=_GroqError, APIStatusError=_GroqError,
        APITimeoutError=_GroqError, AsyncGroq=_UnexpectedGroqClient,
        AuthenticationError=_GroqError, BadRequestError=_GroqError,
        InternalServerError=_GroqError, NotFoundError=_GroqError,
        RateLimitError=_GroqError,
    )

from ai_analyst import AIAnalyst


class BackendSwitchTests(unittest.TestCase):
    def test_missing_backend_setting_defaults_to_local(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = SimpleNamespace(
                data_dir=tmp, groq_daily_token_budget=1000,
                groq_api_key="not-used-in-local-mode", groq_model="openai/gpt-oss-20b",
                http_timeout_sec=2,
            )
            analyst = AIAnalyst(cfg)
            self.assertEqual(analyst.decision_backend, "local")
            asyncio.run(analyst.start())
            self.assertIsNone(analyst._client)
            asyncio.run(analyst.close())

    def test_local_backend_starts_without_groq_client(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DECISION_BACKEND": "local"}):
            cfg = SimpleNamespace(
                data_dir=tmp, groq_daily_token_budget=1000,
                groq_api_key="not-used-in-local-mode", groq_model="openai/gpt-oss-20b",
                http_timeout_sec=2,
            )
            analyst = AIAnalyst(cfg)
            asyncio.run(analyst.start())
            self.assertEqual(analyst.decision_backend, "local")
            self.assertIsNone(analyst._client)
            self.assertEqual(asyncio.run(analyst.analyze_candidates([{
                "ticker": "SBER", "trigger_kind": "micro_impulse",
                "microstructure": {"kind": "micro_impulse", "projected_30s_pct": 0.3},
                "allowed_setups": ["micro_impulse"], "spread_pct": 0.05,
                "round_trip_cost_pct": 0.1, "breakeven_probability": 0.5,
            }], scalp_mode=True))["SBER"]["action"], "hold")
            self.assertIsNone(asyncio.run(analyst.analyze_market("test", scalp_mode=True)))
            answer = asyncio.run(analyst.answer_question("status?", ""))
            self.assertIn("DECISION_BACKEND=local", answer)
            asyncio.run(analyst.close())


if __name__ == "__main__":
    unittest.main()
