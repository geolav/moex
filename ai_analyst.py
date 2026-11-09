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
- торговые решения принимаются только из валидного JSON (иначе HOLD);
- максимум ~500 токенов на ответ - экономия лимитов TPD.

Модуль потребляет ~10 МБ ОЗУ (httpx под капотом), клиент создаётся один раз.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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
    "'Unclosed connector', TelegramNetworkError / ServerDisconnectedError "
    "('Failed to fetch updates') - следы остановки/рестарта процесса и "
    "обрыва long-poll, норма;\n"
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

# Промпт "собеседника": владелец задаёт короткие вопросы ИИ-трейдеру (/ask).
_ASK_SYSTEM_PROMPT = (
    "Ты - ИИ-трейдер, который управляет этим торговым ботом на Мосбирже "
    "(стратегия - возврат к среднему: покупка при откате цены ниже средних, "
    "продажа по стоп-лоссу/тейк-профиту). Тебе дают текущую сводку по счёту, "
    "бумагам и твоим последним решениям, а также вопрос владельца.\n"
    "ПРАВИЛА: отвечай по-русски, кратко (до 100 слов), человеческим языком, "
    "термины объясняй; опирайся ТОЛЬКО на данные из сводки; не выдумывай "
    "цифр; ссылайся на конкретные значения из сводки (порог сигнала, оценка "
    "score, лимиты, кулдаун) - владелец хочет понять, ПОЧЕМУ сделка прошла "
    "или не прошла; если данных не хватает - прямо скажи, чего именно "
    "не хватает. Отвечай сразу финальным текстом, без служебных рассуждений."
)

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# Промпт "скаута рынка" (scanner.py): один запрос на весь список кандидатов.
# v1.7.2: помимо вердиктов по каждому тикеру просим ОДНО короткое резюме
# простыми словами (что советует покупать и в каких количествах) - владелец
# просил, чтобы список шёл по убыванию интересности, а в конце был короткий
# человеческий совет без сложных терминов.
_SCAN_SYSTEM_PROMPT = (
    "Ты - скаут фондового рынка Мосбиржи. Тебе дают таблицу акций-кандидатов: "
    "цена, индикаторный сигнал от -1 до +1 (чем выше, тем сильнее аргумент за "
    "отскок после отката), RSI, изменение за день, отклонение от рабочей "
    "средней, суточный оборот.\n"
    "ЗАДАЧИ:\n"
    "1) Для КАЖДОГО тикера дай короткий вердикт по-русски (до 12 слов): стоит "
    "ли рассматривать покупку сейчас или лучше пропустить и почему "
    "(перепроданность и затухание падения - за; перекупленность, падение "
    "продолжается - против). Опирайся ТОЛЬКО на таблицу, не выдумывай фактов.\n"
    "2) В конце дай ОДНО резюме простыми словами (до 40 слов, без сложных "
    "терминов): какие 1-3 бумаги сейчас самые интересные для покупки и "
    "сколько примерно лотов (по 1-3) в каждую стоит смотреть. Если покупать "
    "сейчас ничего не стоит - прямо скажи об этом.\n"
    "ОТВЕТ - СТРОГО один JSON-объект без пояснений, формат:\n"
    '{"verdicts": {"TICKER1": "вердикт", "TICKER2": "вердикт"}, '
    '"advice": "короткое резюме простыми словами"}'
)

# Промпт "трейдера". v1.7.2 - МАКСИМАЛЬНО ПОДРОБНАЯ база знаний: вшита
# выжимка открытых трейдер-скиллов, которые скидывал владелец (ссылки в
# промпт не вставляем - LLM не открывает URL; знания вшиты текстом):
#
# tradermonty/claude-trading-skills (eq-инвестиции, свинг):
#   - technical-analyst: тренд/поддержки-сопротивления/объёмы, разработка
#     сценариев (база/бык/медведь с вероятностями и уровнями отмены);
#   - position-sizer: риск на сделку 1% (макс 2%), портфельный риск 6-8%,
#     стоп по ATR, самый строгий лимит - главный;
#   - pre-trade-discipline-gate: нет плана - нет сделки; окно реванша после
#     убытка; недостающие данные = hold;
#   - breakout-trade-planner/vcp-screener: покупать пробой, а не диапазон.
#
# agiprolabs/claude-trading-skills (количественные методы):
#   - mean-reversion: Hurst/z-score (вход |z|>2, выход 0, стоп 3), half-life;
#   - volatility-modeling: EWMA lambda=0.94, размер позиции ~ 1/волатильность;
#   - exit-strategies: лестница тейков 25% траншами, безубыток после первого,
#     трейлинг ATR/EMA, временной стоп, выход по RSI>70/MACD-кроссу;
#   - walk-forward-validation: не доверять правилу, сработавшему 1 раз.
#
# Плюс наш прогноз цены (predictor.py) и стратегия возврата к среднему.
_TRADER_SYSTEM_PROMPT = (
    "Ты - опытный трейдер Мосбиржи; управляешь торговым ботом со стратегией "
    "возврата к среднему на горизонте от часа до суток. Тебе дают сводку по "
    "одной акции: рабочая средняя (SMA) и медленная SMA, RSI, MACD, полосы "
    "Боллинджера, изменения цены за 1/6/24 ч, волатильность, свежий путь "
    "цены, статистический ПРОГНОЗ, новости и твои прошлые решения.\n"
    "КАК АНАЛИЗИРОВАТЬ (внутренне, не печатай):\n"
    "1) Тренд: цена выше или ниже SMA; быстрая/медленная SMA показывают "
    "направление и силу; наклон средних важен: растущие средние усиливают "
    "аргументы за рост.\n"
    "2) Импульс: MACD-гистограмма и её изменение; RSI (меньше 30 - перепродан, "
    "больше 70 - перекуплен).\n"
    "3) Положение в коридоре: %B Боллинджера (около 0 - у нижней границы, "
    "кандидат на отскок; около 1 - у верхней границы, не покупать на хаях).\n"
    "4) Волатильность: при сильном дрожании цены уменьшай уверенность и "
    "размер мыслимой позиции; в шторм даже хорошая идея ведёт к стопу.\n"
    "5) Новости: явно плохие новости перебивают технические сигналы.\n"
    "6) Прогноз цены: используй как ОДИН из аргументов, не как истину: "
    "если прогноз и индикаторы смотрят в одну сторону - уверенность выше; "
    "если противоречат - держи деньги.\n"
    "7) Память сделок: в сводке есть статистика твоих ПРОШЛЫХ завершённых "
    "сделок по этой бумаге (чем кончались, в какие часы). Если похожие "
    "покупки регулярно заканчивались убытком - снижай уверенность или "
    "выбирай hold; если продажа по стоп-лоссу случалась, когда ты отмахивался "
    "от сломанного тренда - не повторяй этой ошибки.\n"
    "8) Дисциплина: если аргументы противоречивы - держи деньги (hold); "
    "не выдумывай чисел, которых нет в сводке.\n"
    "БАЗА ЗНАНИЙ ТРЕЙДЕРА (выжимка открытых скиллов и классики, учитывай всегда):\n"
    "ВХОД:\n"
    "- Не гнаться за ценой: покупать у локального максимума - худшая сделка. "
    "Если 'положение в диапазоне' >= 80% или до максимума меньше 0.5% - это "
    "погоня: hold и жди отката глубже. Покупать пробой уровня с объёмом - "
    "можно, но не вплотную к хаю диапазона.\n"
    "- Первый откат от тренда редко бывает последним: лучше пропустить "
    "первый 'дешёвый' вход, чем купить в начале волнового падения. "
    "Падение, которое ускоряется, не покупают: ловить падающий нож плохо, "
    "жди затухания импульса вниз (свечи мельче, объёмы падают).\n"
    "- Возврат к среднему работает, когда рынок в боковике: z-отклонение от "
    "средней больше 2 - кандидат на отскок, у средней (z около 0) - выход; "
    "если |z| дошёл до 3 - идея отменяется, это не откат, а слом.\n"
    "- В тренде (цена держится выше средней, средние ряд друг над другом) "
    "откаты к средней покупаются, а 'дешевизна' у нижней полосы - не аргумент.\n"
    "- Объёмы подтверждают движение: рост на падающих объёмах не доверяют; "
    "всплеск объёма у уровня - сильный аргумент. У бумаг с маленьким "
    "оборотом сигналы врут - снижай уверенность.\n"
    "РИСК:\n"
    "- Риск на сделку ~1% портфеля, максимум 2%. Суммарный открытый риск по "
    "всем позициям - не выше 6-8%. Самый строгий из лимитов - главный.\n"
    "- Стоп ставь по волатильности (примерно 2 ATR от входа), а не «где "
    "удобно»: в тихой бумаге он ближе, в дёрганой - дальше. Убыток глубже "
    "стопа - запрещён: не советуй держать позицию ниже стоп-уровня без явных "
    "признаков восстановления.\n"
    "- После убытка не мстить рынку той же бумагой сразу: окно реванша - "
    "пауза и повышенные требования.\n"
    "ВЫХОД:\n"
    "- Прибыль фиксируют лестницей, а не всё сразу: часть - у первой цели, "
    "стоп в безубыток после первого транша, дальше letting run с трейлингом.\n"
    "- Сильный рост + затухающий импульс = разворот ближе, чем кажется. "
    "Трейлинг: от пика теряем не больше пары десятых процента (умная "
    "фиксация движка делает это сама - не советуй держать через неё). "
    "Прибыль не бывает 'слишком ранней': пропуск отката хуже маленького заработка.\n"
    "- Время тоже аргумент: внутридневная сделка, за несколько часов не "
    "пошедшая в плюс, - кандидат на закрытие по времени.\n"
    "- RSI выше 70 и медвежий кросс MACD - классические сигналы выхода, "
    "даже если цель ещё не достигнута.\n"
    "- Прогноз конуса: если даже верх 80% конуса едва покрывает издержки, "
    "соотношение риск/прибыль плохое - лучше hold.\n"
    "САМОКОНТРОЛЬ:\n"
    "- Правило, сработавшее один раз, - не закон: не переоценивай "
    "единичный паттерн из памяти сделок.\n"
    "- Утром (в сводке есть блок УТРО) ночных торгов не было: не делай "
    "выводов из вечерней сессии и 'ночного графика'. Опирайся на вчерашнюю "
    "основную сессию и её закрытие; утренние движения часто НЕ продолжают "
    "вчерашний вечерний тренд.\n"
    "- Если данных не хватает или картина противоречива - hold безопаснее "
    "плохой сделки; деньги в кэше - тоже позиция.\n"
    "ПОРЯДОК МЫСЛЕЙ (внутренне): аргументы 'быка' (за рост) -> аргументы "
    "'медведя' (за падение) -> взвесь их с учётом горизонта стратегии и "
    "прогноза. Если позиция уже открыта - оцени, стоит ли закрыть её раньше "
    "цели (sell) или оставить (hold); без позиции - buy или hold.\n"
    "ОТВЕТ - СТРОГО один JSON-объект без пояснений, формат:\n"
    '{"action": "buy|sell|hold", "confidence": 0.0..1.0, '
    '"reason": "одно короткое предложение по-русски, человеческим языком"}'
)


def _load_prompt_file(name: str, fallback: str = "") -> str:
    """Загружает редактируемый system prompt из ./prompts; fallback встроен."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
        return value or fallback
    except OSError:
        log.warning("Файл промпта %s не найден; используется встроенный вариант", path)
        return fallback


_TRADER_SYSTEM_PROMPT = _load_prompt_file("trader_system_ru.md", _TRADER_SYSTEM_PROMPT)
_SCALP_SYSTEM_PROMPT = _load_prompt_file("scalp_system_ru.md")
_AUDITOR_SYSTEM_PROMPT = _load_prompt_file("trade_auditor_ru.md", _AUDIT_SYSTEM_PROMPT)


def parse_decision_json(raw_text: str) -> Optional[dict]:
    """Строго разбирает JSON-вердикт. Текстовый фолбэк запрещён."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if not (0 <= start < end):
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in ("buy", "sell", "hold"):
        return None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None
    reason = str(data.get("reason", "")).strip()
    if not reason:
        return None
    if action == "buy":
        setup = str(data.get("setup", "")).strip().lower()
        if setup not in {"pullback", "breakout", "reversal"}:
            return None
        try:
            stop_pct = float(data.get("stop_pct"))
            tp1_pct = float(data.get("tp1_pct"))
        except (TypeError, ValueError):
            return None
        if not (0.0 < stop_pct <= 5.0 and 0.0 < tp1_pct <= 15.0):
            return None
        size_mult = data.get("size_mult", 1.0)
        if size_mult not in (0.5, 1, 1.0):
            return None
        data["setup"] = setup
        data["stop_pct"] = stop_pct
        data["tp1_pct"] = tp1_pct
        data["size_mult"] = float(size_mult)
    data["action"] = action
    data["confidence"] = confidence
    data["reason"] = reason[:300]
    return data


#: Коды HTTP, при которых имеет смысл повторить запрос (5xx).
_RETRYABLE_STATUS_MIN = 500


def _parse_scan_answer(raw_text: str) -> Tuple[Dict[str, str], str]:
    """Разбор ответа скаута: (вердикты по тикерам, резюме простыми словами).

    v1.7.2: формат {"verdicts": {...}, "advice": "..."}; понимаем и старый
    плоский {ticker: вердикт}. Если JSON разбит мусором - вытаскиваем пары
    "TICKER": "вердикт" регуляркой, чтобы комментарии не пропадали совсем
    (раньше один битый ответ убивал весь блок ИИ в отчёте сканера).
    """
    text = raw_text or ""
    verdicts: Dict[str, str] = {}
    advice = ""
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            inner = data.get("verdicts")
            if isinstance(inner, dict):
                for k, v in inner.items():
                    if str(v).strip():
                        verdicts[str(k).strip().upper()] = str(v).strip()[:120]
                advice = str(data.get("advice", "") or "").strip()[:400]
            else:
                for k, v in data.items():
                    if str(v).strip() and k != "advice":
                        verdicts[str(k).strip().upper()] = str(v).strip()[:120]
                advice = str(data.get("advice", "") or "").strip()[:400]
    if not verdicts:
        # Фолбэк: пары "TICKER": "текст" даже из битого JSON.
        for m in re.finditer(r'"([A-ZА-ЯЁ]{3,6}[A-Z0-9]*)"?\s*:\s*"([^"]{4,120})"',
                             text):
            key, val = m.group(1).strip().upper(), m.group(2).strip()
            if key in ("VERDICTS", "ADVICE"):
                continue
            verdicts.setdefault(key, val[:120])
    if not advice:
        m = re.search(r'"advice"\s*:\s*"([^"]{4,400})"', text)
        if m:
            advice = m.group(1).strip()[:400]
    if not verdicts and not advice:
        log.warning("Сканер: ИИ ответил без узнаваемого формата - %.120r", text)
    return verdicts, advice


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

        model = self._cfg.groq_model
        # gpt-oss сначала "думает" - reasoning-токены расходуют лимит ответа,
        # и при маленьком max_tokens content приходит ПУСТЫМ (finish_reason
        # ='length'), а бот показывал заглушку 'Не смог сформулировать ответ'.
        # Просим низкие рассуждения точечно: llama такой параметр не знает
        # и вернул бы 400.
        extra: Dict[str, str] = {}
        if "gpt-oss" in (model or "").lower():
            extra["reasoning_effort"] = "low"

        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        last_exc: Optional[Exception] = None
        empty_streak = 0
        tokens_eff = max_tokens

        for attempt in range(1, tries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=tokens_eff,
                    **extra,
                )
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                if content:
                    log.debug("Groq ответил за попытку %d: %.120r", attempt, content)
                    return content

                # Пустой content: весь лимит токенов ушёл на рассуждения.
                # На следующей попытке даём в 2.5 раза больше токенов и
                # требуем финальный ответ сразу.
                empty_streak += 1
                reasoning = (getattr(choice.message, "reasoning", "") or "").strip()
                log.warning(
                    "Groq вернул пустой content (попытка %d/%d, finish_reason=%s, "
                    "reasoning=%d симв., лимит был %d ток.)",
                    attempt, tries, getattr(choice, "finish_reason", "?"),
                    len(reasoning), tokens_eff)
                # Поле reasoning не является торговым ответом и не должно
                # никогда подменять content: при пустом content продолжаем
                # retry, затем analyze_market вернёт HOLD (fail-closed).
                tokens_eff = int(tokens_eff * 2.5) + 100
                if len(messages) == 2:
                    messages.append({
                        "role": "system",
                        "content": "Финальный ответ обязателен: сразу дай ответ, "
                                   "без внутренних рассуждений.",
                    })

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

        if last_exc is None:
            raise AnalystError(
                f"Модель {model} три раза вернула пустой ответ "
                f"(лимит последней попытки: {tokens_eff} ток.) - попробуйте позже")
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
        # max_tokens 384: gpt-oss тратит часть на рассуждения - раньше лимит
        # 16 токенов давал пустой ответ, и сентимент молча считался нейтральным.
        snippet = news_text.strip()[:1500]
        try:
            raw = await self._chat(
                _SENTIMENT_SYSTEM_PROMPT,
                snippet,
                max_tokens=384,
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
            _AUDITOR_SYSTEM_PROMPT,
            f"{_AUDIT_USER_PROMPT}\n\nЛоги:\n{snippet}",
            max_tokens=350,
            temperature=0.2,
        )
        return raw.strip() or "Все ОК"

    async def analyze_market(self, market_brief: str,
                             scalp_mode: bool = False) -> Optional[dict]:
        """Вердикт трейдера. Ошибка/невалидный JSON -> None (движок HOLD)."""
        brief = (market_brief or "").strip()[:7000]
        if not brief:
            return None
        system_prompt = _TRADER_SYSTEM_PROMPT
        if scalp_mode and _SCALP_SYSTEM_PROMPT:
            system_prompt += "\n\n" + _SCALP_SYSTEM_PROMPT
        try:
            raw = await self._chat(
                system_prompt, brief,
                max_tokens=500, temperature=0.1,
            )
        except AnalystError as exc:
            log.warning("Трейдер-LLM недоступен (%s) - fail-closed: HOLD", exc)
            return None
        decision = parse_decision_json(raw)
        if not decision:
            log.warning("Трейдер-LLM вернул невалидный JSON; fail-closed: HOLD: %.120r", raw)
            return None
        return {
            "action": decision["action"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "setup": str(decision.get("setup", "none"))[:32],
            "stop_pct": float(decision.get("stop_pct", 0) or 0),
            "tp1_pct": float(decision.get("tp1_pct", 0) or 0),
            "size_mult": 0.5 if decision.get("size_mult") == 0.5 else 1.0,
        }

    async def scan_verdicts(self, rows: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Короткие вердикты ИИ по кандидатам сканера рынка (ОДНИМ запросом).

        rows: [{ticker, price, score, rsi, day_chg_pct, vs_sma_pct, turnover_txt}].
        Возвращает {ticker: вердикт}. Любая ошибка/нечитаемый ответ -> {}
        (сканер просто опустит комментарии - торговля не зависит от ИИ).

        v1.7.2: оставлен для совместимости - сканер вызывает
        scan_verdicts_with_advice (он же даёт короткое резюме для отчёта).
        """
        verdicts, _advice = await self.scan_verdicts_with_advice(rows)
        return verdicts

    async def scan_verdicts_with_advice(
            self, rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], str]:
        """
        Вердикты по кандидатам + короткое резюме простыми словами (v1.7.2).

        Возвращает (verdicts, advice): verdicts - {ticker: вердикт},
        advice - одно резюме по списку (что советует покупать и сколько
        лотов, простыми словами) или '' - если ИИ не дал.
        Разбор ответа устойчив к мусору: JSON -> фолбэк-регулярки.
        """
        if not rows:
            return {}, ""
        lines = []
        for r in rows:
            lines.append(
                f"{r.get('ticker')}: цена {float(r.get('price') or 0):.2f} ₽, "
                f"сигнал {float(r.get('score') or 0):+.2f}, "
                f"RSI {float(r.get('rsi') or 0):.0f}, "
                f"за день {float(r.get('day_chg_pct') or 0):+.1f}%, "
                f"к средней {float(r.get('vs_sma_pct') or 0):+.1f}%, "
                f"оборот {r.get('turnover_txt', '?')}")
        try:
            raw = await self._chat(
                _SCAN_SYSTEM_PROMPT,
                "Таблица кандидатов:\n" + "\n".join(lines),
                max_tokens=900,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - ИИ не обязателен
            log.info("Сканер: ИИ-комментарии недоступны (%s)", exc)
            return {}, ""
        return _parse_scan_answer(raw)

    async def answer_question(self, question: str, context: str) -> str:
        """
        Ответ на вопрос владельца (/ask): сводка по счёту/бумагам + вопрос.

        При ошибке API бросает AnalystError - обработчик /ask покажет её
        пользователю. Ответ обрезан до ~400 токенов: краткий человеческий
        ответ, не эссе.
        """
        q = (question or "").strip()[:500]
        ctx_text = (context or "").strip()[:6000]  # страховка TPM free-тарифа
        if not q:
            return "Пустой вопрос - нечего отвечать."
        raw = await self._chat(
            _ASK_SYSTEM_PROMPT,
            f"Сводка:\n{ctx_text}\n\nВопрос владельца: {q}",
            max_tokens=800,
            temperature=0.4,
        )
        return raw.strip() or "Не смог сформулировать ответ - попробуй переформулировать вопрос."


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
