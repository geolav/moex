"""
ai_analyst.py - ИИ-аналитик на бесплатном Groq API.

Две функции:
  1. analyze_sentiment(news_text) -> float от -1.0 до 1.0
     (тональность новостей для торговой стратегии);
  2. audit_logs(log_text) -> краткий вердикт для ИИ-аудитора логов.

Особенности:
- модель по умолчанию: llama-3.1-8b-instant (free-тариф Groq: 30 RPM /
  14 400 RPD; старая llama3-8b-8192 выведена из эксплуатации Groq);
- все запросы с retry и экспоненциальной задержкой; при RateLimitError
  уважаем заголовок retry-after;
- ответы парсятся устойчиво к мусору (ищем первое число в тексте);
- максимум ~500 токенов на ответ - экономия лимитов TPD.

Модуль потребляет ~10 МБ ОЗУ (httpx под капотом), клиент создаётся один раз.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

log = logging.getLogger("ai_analyst")


class AnalystError(RuntimeError):
    """Невосполнимая ошибка ИИ-аналитика (неверный ключ, плохой запрос)."""


_SENTIMENT_SYSTEM_PROMPT = (
    "Ты - финансовый аналитик российской фондовой биржи MOEX. "
    "Тебе дают выдержку из свежих новостей, касающихся конкретной акции. "
    "Оцени, как эти новости повлияют на цену акции в ближайшие дни. "
    "Ответь СТРОГО одним числом от -1.0 до 1.0: "
    "-1.0 = крайне негативные новости, 0.0 = нейтральные/нет значимых новостей, "
    "1.0 = крайне позитивные. Без слов, без пояснений, только число."
)

# Промпт аудитора - дословно по требованиям заказчика + минимальная обвязка.
_AUDIT_SYSTEM_PROMPT = (
    "Ты - DevOps-инженер, следящий за торговым ботом на Python. "
    "Отвечай кратко, по-русски, без лишних рассуждений."
)
_AUDIT_USER_PROMPT = (
    "Проанализируй эти логи торгового бота. Есть ли критические ошибки, "
    "аномалии или признаки сбоя? Ответь кратко: 'Все ОК' или опиши проблему."
)

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

#: Коды HTTP, при которых имеет смысл повторить запрос (5xx).
_RETRYABLE_STATUS_MIN = 500


def parse_sentiment_float(raw_text: str) -> float:
    """
    Извлекает число -1.0..1.0 из ответа модели.

    Модели любят отвечать "0.35 (умеренно позитивно)" - берём первое число.
    Запятая как десятичный разделитель тоже понимается. При полной неудаче 0.0.
    """
    match = _NUM_RE.search(raw_text or "")
    if not match:
        return 0.0
    value = float(match.group(0).replace(",", "."))
    return max(-1.0, min(1.0, value))


class AIAnalyst:
    """Обёртка над AsyncGroq с retry-логикой и двумя задачами."""

    def __init__(self, config) -> None:
        self._cfg = config
        self._client: Optional[AsyncGroq] = None

    # --- жизненный цикл ------------------------------------------------------
    async def start(self) -> None:
        # max_retries=0: ретраи делаем сами (контроль задержек и логов).
        self._client = AsyncGroq(
            api_key=self._cfg.groq_api_key,
            timeout=self._cfg.http_timeout_sec,
            max_retries=0,
        )
        log.info("AIAnalyst готов (модель %s)", self._cfg.groq_model)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # --- низкоуровневый вызов с retry -----------------------------------------
    async def _chat(self, system: str, user: str, *, max_tokens: int, temperature: float,
                    tries: int = 3) -> str:
        if self._client is None:
            raise AnalystError("AIAnalyst не запущен (вызовите start())")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        last_exc: Exception | None = None

        for attempt in range(1, tries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._cfg.groq_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                log.debug("Groq ответил за попытку %d: %.120r", attempt, content)
                return content

            except AuthenticationError as exc:
                # Ключ неверный - повторять бессмысленно.
                raise AnalystError(f"Groq: неверный API-ключ: {exc}") from exc

            except (BadRequestError, NotFoundError) as exc:
                raise AnalystError(f"Groq: плохой запрос/модель ({exc})") from exc

            except RateLimitError as exc:
                # Лимит free-тарифа: жду сколько велит retry-after, иначе дольше обычного.
                retry_after = _extract_retry_after(exc)
                wait = max(retry_after or 0.0, delay * 4.0)
                last_exc = exc
                log.warning("Groq RateLimit (попытка %d/%d), жду %.0f сек: %s",
                            attempt, tries, wait, exc)
                await asyncio.sleep(wait)

            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                last_exc = exc
                log.warning("Groq сеть/сервер (попытка %d/%d): %s", attempt, tries, exc)
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                status = getattr(exc, "status_code", 0) or 0
                if status >= _RETRYABLE_STATUS_MIN:
                    last_exc = exc
                    log.warning("Groq HTTP %d (попытка %d/%d)", status, attempt, tries)
                    await asyncio.sleep(delay)
                else:
                    raise AnalystError(f"Groq HTTP {status}: {exc}") from exc

            delay *= 2.0  # экспоненциальная задержка: 2, 4, 8 сек

        raise AnalystError(f"Groq не ответил после {tries} попыток: {last_exc}")

    # --- публичные задачи -------------------------------------------------------
    async def analyze_sentiment(self, news_text: str) -> float:
        """
        Тональность новостей: float от -1.0 до 1.0.
        Пустой текст -> 0.0 (нейтрально), без обращения к API (экономия лимитов).
        """
        if not news_text or not news_text.strip():
            return 0.0
        # ~1500 символов хватит с запасом: экономим TPM free-тарифа (6K TPM).
        snippet = news_text.strip()[:1500]
        try:
            raw = await self._chat(
                _SENTIMENT_SYSTEM_PROMPT,
                snippet,
                max_tokens=16,
                temperature=0.0,
            )
        except AnalystError as exc:
            log.error("Сентимент недоступен (%s); возвращаю нейтральный 0.0", exc)
            return 0.0
        return parse_sentiment_float(raw)

    async def audit_logs(self, log_text: str) -> str:
        """Вердикт аудитора: короткий текст ('Все ОК' или описание проблемы)."""
        if not log_text.strip():
            return "Лог пуст - анализировать нечего."
        snippet = log_text.strip()[-8000:]  # страховка по размеру запроса
        raw = await self._chat(
            _AUDIT_SYSTEM_PROMPT,
            f"{_AUDIT_USER_PROMPT}\n\nЛоги:\n{snippet}",
            max_tokens=200,
            temperature=0.2,
        )
        return raw.strip() or "Все ОК"


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Достаёт Retry-After из ответа Groq, если заголовок присутствует."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Функция уровня модуля (сигнатура из ТЗ: analyze_sentiment(news_text) -> float).
# Использует глобальный экземпляр, который инициализирует main.py.
# ---------------------------------------------------------------------------
_default_analyst: Optional[AIAnalyst] = None


def set_default_analyst(analyst: AIAnalyst) -> None:
    global _default_analyst
    _default_analyst = analyst


async def analyze_sentiment(news_text: str) -> float:
    """Модульный вход: analyze_sentiment(news_text: str) -> float."""
    if _default_analyst is None:
        raise AnalystError("Аналитик не инициализирован (set_default_analyst не вызван)")
    return await _default_analyst.analyze_sentiment(news_text)
