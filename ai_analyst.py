"""
ai_analyst.py — компактный слой Groq для финального решения по MOEX.

В штатной автоторговле вызывается один раз на скан: получает до трёх
кандидатов, уже отобранных локальной математикой и прошедших OHLCV-фильтр.
Новости, рейтинг, сетапы, sizing, стопы, выходы и аудит работают локально.
Ручной /ask остаётся отдельным пользовательским вызовом.

Защита квоты: персистентный дневной token budget, резерв до запроса,
учёт фактического usage, ограничение частоты пакетных запросов, короткий
JSON-ответ, низкий reasoning_effort для GPT-OSS и жёсткий
max_completion_tokens. Сбой/исчерпание бюджета = fail-closed (без покупки).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
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

_CANDIDATE_BATCH_SYSTEM_PROMPT = ""

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
_CANDIDATE_BATCH_SYSTEM_PROMPT = _load_prompt_file("candidate_batch_ru.md")
_SCALP_CANDIDATE_BATCH_SYSTEM_PROMPT = _load_prompt_file(
    "scalp_candidate_batch_ru.md", _CANDIDATE_BATCH_SYSTEM_PROMPT)


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


class DailyTokenBudget:
    """Персистентный локальный предохранитель от исчерпания Groq TPD.

    Счётчик фактического usage сохраняется между перезапусками. Для запроса
    заранее резервируется консервативная оценка prompt + completion; после
    ответа резерв заменяется реальным total_tokens из ответа Groq.
    """

    def __init__(self, path: str, limit: int) -> None:
        self.path = Path(path)
        self.limit = max(1, int(limit))
        self._lock = asyncio.Lock()
        self._pending = 0
        self._day = datetime.now(timezone.utc).date().isoformat()
        self._used = 0
        self._blocked_until = 0.0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date_utc") == self._day:
                self._used = max(0, int(data.get("used_tokens", 0)))
                # Если процесс оборвался во время API-запроса, считаем сохранённый
                # резерв израсходованным: запрос мог успеть попасть в Groq.
                interrupted_reservation = max(0, int(data.get("reserved_tokens", 0)))
                self._used += interrupted_reservation
                self._blocked_until = max(0.0, float(data.get("blocked_until", 0) or 0))
                if interrupted_reservation:
                    self._save()
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("Счётчик Groq token budget не прочитан (%s); начинаю локальный учёт с 0", exc)

    def _rollover(self) -> None:
        day = datetime.now(timezone.utc).date().isoformat()
        if day != self._day:
            self._day = day
            self._used = 0
            self._pending = 0
            self._blocked_until = 0.0
            self._save()

    def _save(self, model: str = "") -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "date_utc": self._day,
                "used_tokens": int(self._used),
                "reserved_tokens": int(self._pending),
                "daily_budget": int(self.limit),
                "blocked_until": float(self._blocked_until),
                "model": model,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            log.error("Не удалось сохранить счётчик Groq token budget: %s", exc)

    async def reserve(self, estimated_tokens: int) -> Tuple[bool, str]:
        estimated = max(1, int(estimated_tokens))
        async with self._lock:
            self._rollover()
            now = __import__("time").time()
            if self._blocked_until > now:
                return False, f"Groq временно заблокирован ещё {int(self._blocked_until - now)} сек"
            available = self.limit - self._used - self._pending
            if estimated > available:
                return False, (f"локальный дневной бюджет Groq: использовано {self._used}, "
                               f"зарезервировано {self._pending}, запрос ≈{estimated}, "
                               f"лимит {self.limit} токенов")
            self._pending += estimated
            # Фиксируем резерв ДО отправки запроса, чтобы рестарт не обнулил
            # потенциально уже потраченные токены.
            self._save()
            return True, ""

    async def settle(self, reserved_tokens: int, actual_tokens: int, model: str) -> None:
        async with self._lock:
            self._rollover()
            self._pending = max(0, self._pending - max(0, int(reserved_tokens)))
            self._used += max(0, int(actual_tokens))
            self._save(model)
            log.info("Groq token budget: запрос %d токенов; сегодня %d/%d (резерв %.0f%%)",
                     max(0, int(actual_tokens)), self._used, self.limit,
                     max(0.0, (1.0 - self.limit / 200000.0) * 100.0))

    async def release(self, reserved_tokens: int) -> None:
        async with self._lock:
            self._pending = max(0, self._pending - max(0, int(reserved_tokens)))
            self._save()

    async def charge_estimate(self, reserved_tokens: int, model: str) -> None:
        """Консервативно учитывает резерв, если API мог обработать запрос,
        но клиент не получил надёжного ответа (timeout/обрыв/HTTP 5xx)."""
        async with self._lock:
            self._pending = max(0, self._pending - max(0, int(reserved_tokens)))
            self._used += max(0, int(reserved_tokens))
            self._save(model)
            log.warning("Groq ответ не подтверждён; резерв %d токенов учтён как расход", reserved_tokens)

    async def block(self, seconds: float, model: str) -> None:
        async with self._lock:
            self._rollover()
            now = __import__("time").time()
            self._blocked_until = max(self._blocked_until, now + max(30.0, seconds))
            self._save(model)

    def status(self) -> Dict[str, int]:
        self._rollover()
        return {"used_tokens": self._used, "budget_tokens": self.limit,
                "remaining_tokens": max(0, self.limit - self._used - self._pending)}


class AIAnalyst:
    """Обёртка над AsyncGroq с общим дневным бюджетом токенов."""

    def __init__(self, config) -> None:
        self._cfg = config
        self._client: Optional[AsyncGroq] = None
        data_dir = getattr(config, "data_dir", os.path.join(os.path.dirname(__file__), "data"))
        usage_path = os.path.join(data_dir, "groq_usage.json")
        self._budget = DailyTokenBudget(usage_path,
                                        getattr(config, "groq_daily_token_budget", 120000))
        self._candidate_call_lock = asyncio.Lock()
        self._last_candidate_call_monotonic = 0.0
        self._last_scalp_candidate_call_monotonic = 0.0

    # --- жизненный цикл ------------------------------------------------------
    async def start(self) -> None:
        # max_retries=0: ретраи и бюджет контролируются здесь, а не SDK.
        base_url = os.getenv("GROQ_BASE_URL", "").strip().rstrip("/") or None
        if base_url:
            log.info("Groq API работает через прокси: %s", base_url)
        self._client = AsyncGroq(
            api_key=self._cfg.groq_api_key,
            base_url=base_url,
            timeout=self._cfg.http_timeout_sec,
            max_retries=0,
        )
        usage = self._budget.status()
        log.info("AIAnalyst готов (модель %s; бюджет Groq %d/%d токенов уже использован)",
                 self._cfg.groq_model, usage["used_tokens"], usage["budget_tokens"])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _estimate_tokens(system: str, user: str, max_tokens: int) -> int:
        # Консервативная оценка для русского текста + запас на reasoning.
        prompt_estimate = math.ceil((len(system) + len(user)) / 2.7 * 1.15)
        completion_estimate = math.ceil(max_tokens * 1.25)
        return max(100, prompt_estimate + completion_estimate)

    # --- низкоуровневый вызов -------------------------------------------------
    async def _chat(self, system: str, user: str, *, max_tokens: int,
                    temperature: float, tries: int = 2,
                    response_format: Optional[Dict[str, Any]] = None) -> str:
        if self._client is None:
            raise AnalystError("AIAnalyst не запущен (вызовите start())")

        model = self._cfg.groq_model
        extra: Dict[str, Any] = {}
        if "gpt-oss" in (model or "").lower():
            extra["reasoning_effort"] = "low"
            extra["include_reasoning"] = False
        if response_format is not None:
            extra["response_format"] = response_format
        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        last_exc: Optional[Exception] = None

        for attempt in range(1, max(1, tries) + 1):
            reserved = self._estimate_tokens(system, user, max_tokens)
            allowed, why = await self._budget.reserve(reserved)
            if not allowed:
                raise AnalystError("Groq запрос не отправлен: " + why)
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    **extra,
                )
                usage = getattr(response, "usage", None)
                actual = getattr(usage, "total_tokens", None) if usage is not None else None
                if actual is None and usage is not None:
                    actual = ((getattr(usage, "prompt_tokens", 0) or 0)
                              + (getattr(usage, "completion_tokens", 0) or 0))
                await self._budget.settle(reserved, int(actual or reserved), model)

                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                if content:
                    return content
                finish = getattr(choice, "finish_reason", "unknown")
                raise AnalystError(
                    f"Groq вернул пустой content (finish_reason={finish}); "
                    "повтор не выполняю, чтобы не расходовать квоту")

            except RateLimitError as exc:
                await self._budget.release(reserved)
                message = str(exc)
                retry_after = _extract_retry_after(exc) or 0.0
                if re.search(r"tokens per day|\bTPD\b", message, re.IGNORECASE):
                    cooldown = max(retry_after, 300.0)
                    await self._budget.block(cooldown, model)
                    raise AnalystError(
                        f"Groq достиг TPD; запросы приостановлены на {int(cooldown)} сек") from exc
                last_exc = exc
                if attempt >= max(1, tries):
                    break
                wait = max(retry_after, delay)
                log.warning("Groq rate limit (не TPD), повтор через %.0f сек", wait)
                await asyncio.sleep(wait)
                delay *= 2.0

            except (AuthenticationError, BadRequestError, NotFoundError) as exc:
                await self._budget.release(reserved)
                raise AnalystError(f"Groq: ошибка конфигурации/запроса: {exc}") from exc

            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                await self._budget.charge_estimate(reserved, model)
                last_exc = exc
                if attempt >= max(1, tries):
                    break
                await asyncio.sleep(delay)
                delay *= 2.0

            except AnalystError:
                # Пустой ответ уже учтён как usage; budget отказов ничего не резервирует.
                raise

            except APIStatusError as exc:
                status = getattr(exc, "status_code", 0) or 0
                if status >= _RETRYABLE_STATUS_MIN:
                    await self._budget.charge_estimate(reserved, model)
                else:
                    await self._budget.release(reserved)
                if status >= _RETRYABLE_STATUS_MIN and attempt < max(1, tries):
                    last_exc = exc
                    await asyncio.sleep(delay)
                    delay *= 2.0
                    continue
                raise AnalystError(f"Groq HTTP {status}: {exc}") from exc

            except Exception:
                # Неизвестная ошибка могла произойти после отправки запроса;
                # удерживаем консервативную оценку, чтобы не превысить TPD.
                await self._budget.charge_estimate(reserved, model)
                raise

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

    def candidate_call_wait_seconds(self, scalp_mode: bool = False) -> float:
        """Сколько осталось до следующего разрешённого пакетного вызова Groq."""
        if scalp_mode:
            minimum = int(getattr(self._cfg, "scalp_ai_min_request_interval_sec", 240) or 240)
            last = self._last_scalp_candidate_call_monotonic
        else:
            minimum = int(getattr(self._cfg, "ai_min_request_interval_sec", 600) or 600)
            last = self._last_candidate_call_monotonic
        return max(0.0, float(minimum) - (time.monotonic() - float(last or 0.0))) if last else 0.0

    async def analyze_candidates(self, rows: List[Dict[str, Any]],
                                 scalp_mode: bool = False) -> Dict[str, dict]:
        """Оценивает до трёх уже отфильтрованных кандидатов одним Groq-запросом."""
        if not rows:
            return {}
        compact: List[Dict[str, Any]] = []
        for row in rows[:3]:
            summary = row.get("summary") or {}
            macd = summary.get("macd") or {}
            bb = summary.get("bollinger") or {}
            fast = summary.get("sma_fast")
            slow = summary.get("sma_slow")
            price = float(row.get("price") or 0.0)
            compact.append({
                "ticker": str(row.get("ticker", ""))[:16],
                "price": round(price, 4),
                "rank": round(float(row.get("rank_score") or 0.0), 1),
                "allowed_setups": list(row.get("allowed_setups") or []),
                "signal": round(float(row.get("score") or 0.0), 3),
                "rsi": round(float(summary.get("rsi14") or 0.0), 1),
                "rsi7": round(float(summary.get("rsi7")), 1)
                    if summary.get("rsi7") is not None else None,
                "ema9_vs_ema21": ("up" if float(summary.get("ema9")) > float(summary.get("ema21"))
                                  else "down")
                    if summary.get("ema9") is not None
                    and summary.get("ema21") is not None else None,
                "bb_squeeze": bool(summary.get("bb_squeeze"))
                    if summary.get("bb_squeeze") is not None else None,
                "macd_hist": round(float(macd.get("histogram") or 0.0), 5),
                "macd_prev": round(float(macd.get("histogram_prev") or 0.0), 5),
                "price_vs_sma_fast_pct": round((price / float(fast) - 1.0) * 100.0, 3)
                    if fast and price > 0 else None,
                "price_vs_sma_slow_pct": round((price / float(slow) - 1.0) * 100.0, 3)
                    if slow and price > 0 else None,
                "chg_1h_pct": round(float(summary.get("chg_1h") or 0.0), 3),
                "volatility_pct_h": round(float(summary.get("volatility") or 0.0), 3),
                "atr_pct": round(float(row.get("atr_pct") or 0.0), 3),
                "planned_stop_pct": round(float(row.get("planned_stop_pct") or 0.0), 3)
                    if row.get("planned_stop_pct") is not None else None,
                "round_trip_cost_pct": round(float(row.get("round_trip_cost_pct") or 0.0), 3)
                    if row.get("round_trip_cost_pct") is not None else None,
                "breakeven_probability": round(float(row.get("breakeven_probability") or 0.0), 3)
                    if row.get("breakeven_probability") is not None else None,
                "spread_pct": round(float(row.get("spread_pct") or 0.0), 3)
                    if row.get("spread_pct") is not None else None,
                "volume_ratio": round(float((row.get("entry_context") or {}).get("volume_ratio") or 0.0), 2),
                "turnover_mln": round(float(row.get("turnover") or 0.0) / 1_000_000.0, 1),
                "day_change_pct": round(float(row.get("day_chg_pct") or 0.0), 2),
                "bollinger_pct_b": round(float(bb.get("pct_b") or 0.0), 2)
                    if bb.get("pct_b") is not None else None,
                # v1.14.0: какой локальный триггер породил кандидата
                # (micro_impulse = импульс потока, dip_reversal = остановившаяся
                # локальная яма) и был ли кандидат отправлен по fallback-
                # правилу (строгий shortlist пуст, ИИ - последний фильтр).
                "trigger": str(row.get("trigger_kind") or "")[:24]
                    if scalp_mode else None,
                "fallback_candidate": bool(row.get("fallback_candidate"))
                    if row.get("fallback_candidate") is not None else None,
                "microstructure": row.get("microstructure") if scalp_mode else None,
            })
        user = json.dumps({
            "mode": "scalp" if scalp_mode else "intraday",
            "market_breadth_pct": self._candidate_breadth(rows),
            "candidates": compact,
        }, ensure_ascii=False, separators=(",", ":"))
        async with self._candidate_call_lock:
            now = time.monotonic()
            if scalp_mode:
                min_interval = int(getattr(
                    self._cfg, "scalp_ai_min_request_interval_sec", 240) or 240)
                last_call = self._last_scalp_candidate_call_monotonic
            else:
                min_interval = int(getattr(self._cfg, "ai_min_request_interval_sec", 600))
                last_call = self._last_candidate_call_monotonic
            elapsed = now - last_call
            if last_call and elapsed < min_interval:
                log.info("Пакетный Groq-запрос %s пропущен: следующий через %.0f сек",
                         "скальпа" if scalp_mode else "сканера", min_interval - elapsed)
                return {}
            if scalp_mode:
                self._last_scalp_candidate_call_monotonic = now
            else:
                self._last_candidate_call_monotonic = now
            try:
                raw = await self._chat(
                    (_SCALP_CANDIDATE_BATCH_SYSTEM_PROMPT if scalp_mode
                     else _CANDIDATE_BATCH_SYSTEM_PROMPT),
                    user,
                    # GPT-OSS расходует часть completion budget на reasoning.
                    # 320 приводило к HTTP 400 json_validate_failed до валидного
                    # JSON. 1024 достаточно для reasoning + 3 компактных решений;
                    # частота пакета (4 мин) и дневной budget не увеличиваются.
                    max_tokens=1024 if scalp_mode else 760,
                    temperature=0.0,
                    tries=1,
                    response_format={"type": "json_object"},
                )
            except AnalystError as exc:
                log.warning("Пакетный ИИ-анализ кандидатов пропущен: %s", exc)
                return {}
        allowed_by_ticker: Dict[str, set] = {}
        fallback_by_ticker: Dict[str, bool] = {}
        for row in rows[:3]:
            ticker = str(row.get("ticker", "")).upper()
            confirmed = set(row.get("allowed_setups") or [])
            trigger = str(row.get("trigger_kind") or "").lower()
            if scalp_mode and trigger in {"micro_impulse", "dip_reversal"}:
                confirmed.add(trigger)
            # В fallback локальный OHLCV-сетап мог не пройти мягкие фильтры;
            # ИИ может назвать reversal, но подтверждение требует повышенной уверенности.
            fallback = bool(row.get("fallback_candidate"))
            if fallback:
                confirmed.add("reversal")
            allowed_by_ticker[ticker] = confirmed
            fallback_by_ticker[ticker] = fallback
        parsed = self._parse_candidate_batch(raw, allowed_by_ticker, fallback_by_ticker)
        log.info("Пакетный ИИ-анализ: обработано %d кандидатов, валидных решений %d",
                 len(compact), len(parsed))
        return parsed

    @staticmethod
    def _candidate_breadth(rows: List[Dict[str, Any]]) -> Optional[float]:
        values = [row.get("market_breadth_pct") for row in rows
                  if row.get("market_breadth_pct") is not None]
        return round(float(values[0]), 1) if values else None

    @staticmethod
    def _parse_candidate_batch(raw: str, allowed_by_ticker: Dict[str, set],
                               fallback_by_ticker: Optional[Dict[str, bool]] = None) -> Dict[str, dict]:
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(decisions, dict):
            return {}
        result: Dict[str, dict] = {}
        allowed_setups = {"pullback", "breakout", "reversal", "micro_impulse",
                          "dip_reversal", "none"}
        fallback_by_ticker = fallback_by_ticker or {}
        for ticker, confirmed_setups in allowed_by_ticker.items():
            item = decisions.get(ticker)
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "hold")).lower()
            if action not in {"buy", "hold"}:
                action = "hold"
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            setup = str(item.get("setup", "none")).lower()
            if setup not in allowed_setups:
                setup = "none"
            # ИИ не может одобрить buy без сетапа и достаточной уверенности.
            min_confidence = 0.70 if fallback_by_ticker.get(ticker, False) else 0.60
            if action == "buy" and (setup == "none" or confidence < min_confidence
                                    or setup not in confirmed_setups):
                action = "hold"
            result[ticker] = {
                "action": action,
                "confidence": confidence,
                "setup": setup,
                "stop_pct": 0.0,
                "tp1_pct": 0.0,
                "size_mult": 1.0,
                "reason": str(item.get("reason", "нет обоснования"))[:120],
            }
        return result

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
            max_tokens=450,
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
