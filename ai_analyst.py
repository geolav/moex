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
import json
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

# Промпт аудитора: кратко и с перечнем известного шума, чтобы ИИ
# не поднимал ложную тревогу на следы рестартов и телеметрию SDK.
_AUDIT_SYSTEM_PROMPT = (
    "Ты - DevOps-инженер, следящий за торговым ботом на Python "
    "(Мосбиржа, T-Invest API, Telegram-уведомления). "
    "Отвечай кратко, по-русски, без лишних рассуждений."
)
_AUDIT_USER_PROMPT = (
    "Проанализируй хвост лога торгового бота.\n\n"
    "ИЗВЕСТНЫЙ ШУМ - НЕ СЧИТАТЬ ПРОБЛЕМОЙ:\n"
    "1) CancelledError, 'Task was destroyed', 'Unclosed client session', "
    "'Unclosed connector' - следы остановки/рестарта процесса, норма;\n"
    "2) SSLCertVerificationError на /api/0/envelope/ и всё про sentry_sdk - "
    "встроенная телеметрия SDK биржи, к торговле отношения не имеет;\n"
    "3) TelegramConflictError - след от запуска второй копии бота в прошлом;\n"
    "4) единичные TIMEOUT/UNAVAILABLE, после которых запросы идут успешно - "
    "сеть моргнула, ретраи сработали.\n\n"
    "РЕАЛЬНЫЕ ПРОБЛЕМЫ - сообщай только о них: сделки не проходят "
    "(повторяющиеся ошибки buy/sell), цикл опроса падает, ошибки API подряд "
    "(неверный токен, доступ запрещён), нехватка памяти, бот завис.\n\n"
    "Ответь кратко: 'Все ОК' - если реальных проблем нет. Если есть - опиши "
    "проблему и укажи, встречается ли она в логе повторно."
)

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# Промпт "трейдера" (скилл из AI-Trader/TradingAgents): мини-дебаты бык/медведь
# внутри одного запроса, затем строгий JSON-вердикт. Экономно по токенам.
_TRADER_SYSTEM_PROMPT = (
    "Ты - опытный трейдер Мосбиржи. Тебе дают сводку по одной акции: "
    "технические индикаторы (средние, RSI, MACD, полосы Боллинджера, изменения "
    "цены, волатильность), свежие новости и твои прошлые решения.\n"
    "ПОРЯДОК МЫСЛЕЙ (внутренне, не печатай): 1) аргументы 'быка' (за рост); "
    "2) аргументы 'медведя' (за падение); 3) взвесь их с учётом того, что "
    "стратегия - возврат к среднему на горизонте нескольких часов-дней.\n"
    "ПРАВИЛА: при явно плохих новостях не покупай; при перепроданности и "
    "стабильных новостях - кандидат на покупку; если позицию выгодно закрыть "
    "до цели - предлагай sell. Не выдумывай чисел, которых нет в сводке.\n"
    "ОТВЕТ - СТРОГО один JSON-объект без пояснений, формат:\n"
    '{"action": "buy|sell|hold", "confidence": 0.0..1.0, '
    '"reason": "одно короткое предложение по-русски, человеческим языком"}'
)


def parse_decision_json(raw_text: str) -> Optional[dict]:
    """
    Достаёт JSON-вердикт трейдера из ответа модели.

    Модели иногда оборачивают JSON в текст/``` - берём от первой '{'
    до последней '}' и парсим. При неудаче ищем действие регуляркой.
    """
    text = raw_text or ""
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    # Фолбэк: ищем слово-действие в тексте.
    lowered = text.lower()
    for action in ("buy", "sell", "hold"):
        if f'"{action}"' in lowered or f"'{action}'" in lowered:
            return {"action": action, "confidence": 0.3, "reason": "ответ без JSON"}
    return None

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
        # GROQ_BASE_URL: если с VPS Groq отвечает 403/блокируется по гео,
        # укажите адрес реверс-прокси (Deno Deploy/Cloudflare Worker) на api.groq.com.
        # ВАЖНО: БЕЗ суффикса /openai/v1 - SDK Groq добавляет его сам!
        #   GROQ_BASE_URL=https://<имя>.deno.dev
        import os as _os

        base_url = _os.getenv("GROQ_BASE_URL", "").strip().rstrip("/") or None
        if base_url:
            log.info("Groq API работает через прокси: %s", base_url)
        self._client = AsyncGroq(
            api_key=self._cfg.groq_api_key,
            base_url=base_url,
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

    async def analyze_market(self, market_brief: str) -> Optional[dict]:
        """
        Скилл 'трейдер': вердикт по сводке рынка -> dict с ключами
        action ('buy'|'sell'|'hold'), confidence (0..1), reason (str).

        При ошибке API возвращает None (движок переходит на правило-фолбэк).
        """
        brief = (market_brief or "").strip()[:6000]  # страховка TPM free-тарифа
        if not brief:
            return None
        try:
            raw = await self._chat(
                _TRADER_SYSTEM_PROMPT,
                brief,
                max_tokens=220,
                temperature=0.2,
            )
        except AnalystError as exc:
            log.warning("Трейдер-LLM недоступен (%s) - фолбэк на правила", exc)
            return None
        decision = parse_decision_json(raw)
        if not decision:
            log.warning("Трейдер-LLM вернул нечитаемый ответ: %.120r", raw)
            return None
        action = str(decision.get("action", "hold")).lower().strip()
        if action not in ("buy", "sell", "hold"):
            action = "hold"
        try:
            confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        reason = str(decision.get("reason", ""))[:300]
        return {"action": action, "confidence": confidence, "reason": reason}


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