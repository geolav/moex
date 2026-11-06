"""
config.py - конфигурация бота и персистентное состояние.

Загружает все параметры из переменных окружения (.env / EnvironmentFile
systemd). Никаких хардкодных токенов - только os.getenv().

Также содержит StateStore: маленькое JSON-хранилище runtime-состояния
(флаги торговли/аудитора, режим песочницы, открытые позиции, последние
действия). Персистентность нужна, чтобы флаги переживали перезапуск
процесса systemd. Файл крошечный (< 5 КБ), атомарная запись через os.replace.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# python-dotenv - опциональная зависимость: под systemd переменные приходят
# через EnvironmentFile, .env нужен в основном для локальных запусков/тестов.
try:
    from dotenv import load_dotenv

    load_dotenv()  # Если .env отсутствует - просто ничего не делает.
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("config")

# Базовый каталог проекта = папка, где лежит этот файл.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Максимум тикеров - жёсткое требование по памяти и лимитам API.
#: v1.6.0: поднято с 5 до 6 - движок ведёт до шести бумаг одновременно.
MAX_TICKERS = 6

_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t", "да"}


class ConfigError(RuntimeError):
    """Критическая ошибка конфигурации (запуск невозможен)."""


# ---------------------------------------------------------------------------
# Небольшие парсеры окружения с безопасными значениями по умолчанию.
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int, *, low: int = 1, high: int = 10**9) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("ENV %s=%r не число, беру значение по умолчанию %s", name, raw, default)
        return default
    return max(low, min(high, value))


def _env_float(name: str, default: float, *, low: float = -1e9, high: float = 1e9) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        log.warning("ENV %s=%r не число, беру значение по умолчанию %s", name, raw, default)
        return default
    return max(low, min(high, value))


def _env_list(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    # dict.fromkeys: убираем дубли, СОХРАНЯЯ порядок из .env
    # (порядок важен: тикеры обрабатываются строго последовательно).
    items = tuple(dict.fromkeys(part.strip().upper() for part in raw.split(",") if part.strip()))
    return items or default


def parse_harvest_tiers(raw: str) -> Tuple[Tuple[float, float], ...]:
    """Парсит HARVEST_TIERS вида "3.0:0.7,5.0:0.4,8.0:0.25".

    Формат пары: пик_прибыли_в_процентах : допустимый_откат_в_пунктах.
    Чем выше пик, тем меньший откат терпим - при сильном росте продаём
    уже при крошечном минусе. Пары сортируются по убыванию пика.
    """
    tiers: List[Tuple[float, float]] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            peak_s, drop_s = part.split(":", 1)
            peak = float(peak_s.strip().replace(",", "."))
            drop = float(drop_s.strip().replace(",", "."))
        except ValueError:
            log.warning("HARVEST_TIERS: не понял пару %r - пропускаю", part)
            continue
        if peak <= 0 or drop <= 0:
            log.warning("HARVEST_TIERS: пара %r должна быть > 0 - пропускаю", part)
            continue
        tiers.append((peak, drop))
    tiers.sort(key=lambda t: t[0], reverse=True)
    return tuple(tiers)


def _env_urls(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """Список URL через запятую - БЕЗ приведения регистра (иначе ломаются ссылки)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


# ---------------------------------------------------------------------------
# Основной объект конфигурации.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BotConfig:
    """Иммутабельный снимок всех настроек бота."""

    # --- Секреты (только из окружения) ---
    tinkoff_token: str            # токен реального счёта
    tinkoff_sandbox_token: str    # токен песочницы
    groq_api_key: str
    telegram_bot_token: str
    telegram_admin_id: int

    # --- Режимы ---
    use_sandbox: bool             # стартовый режим (далее живёт в state.json)
    trading_enabled_default: bool
    auditor_enabled_default: bool

    # --- Стратегия ---
    trading_tickers: Tuple[str, ...]
    sma_period: int
    poll_interval_sec: int
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    sentiment_threshold: float

    # --- ИИ ---
    groq_model: str
    audit_interval_sec: int
    audit_log_lines: int
    sentiment_ttl_sec: int
    llm_decisions_enabled: bool   # скилл 'трейдер-LLM' (AI-Trader-стиль)

    # --- Риск-контроль активности ---
    trade_cooldown_min: int       # пауза между покупками одной бумаги, мин
    max_buys_per_day: int         # потолок покупок в сутки (по всем тикерам)

    # --- Обучение на ошибках (память сделок, trade_memory.py) ---
    learning_enabled: bool        # учитывать итоги прошлых сделок
    memory_max_trades: int        # сколько сделок хранить в памяти
    daily_loss_limit_pct: float   # стоп на день: макс. убыток, % от капитала
    loss_streak_pause_h: float    # пауза бумаге после 3 убытков подряд, ч
    sl_confirm_cycles: int        # циклов ниже стопа до продажи (1 = сразу)
    sl_veto_confidence: float     # уверенность ИИ, чтобы отложить стоп-лосс
    hard_sl_mult: float           # аварийный стоп = стоп-лосс * этот множитель
    trailing_activate_pct: float  # с какой прибыли включается трейлинг-стоп
    trailing_drop_pct: float      # откат от максимума прибыли для продажи
    llm_early_sell_min_profit_pct: float  # ИИ продаёт досрочно только в плюс
    # --- Умная фиксация прибыли (HARVEST_TIERS) -------------------------------
    # Чем выше был пик прибыли, тем меньший откат от него терпим:
    # (пик %, допуск отката в пунктах). Пары отсортированы по убыванию пика.
    harvest_tiers: Tuple[Tuple[float, float], ...]
    # --- Сканер рынка (scanner.py) -------------------------------------------
    scan_enabled: bool            # фоновый поиск кандидатов раз в N минут
    scan_interval_min: int        # период сканирования рынка, минут
    scan_min_turnover: float      # мин. суточный оборот бумаги, млн руб
    scan_min_price: float         # отсечь «копеечные» бумаги, руб
    scan_deep_n: int              # сколько бумаг глубоко считать по свечам
    scan_report_top: int          # сколько кандидатов присылать в отчёте
    scan_llm: bool                # ИИ-комментарий по кандидатам (1 запрос)

    # --- Инфраструктура ---
    rate_limit_rpm: int
    news_rss_urls: Tuple[str, ...]
    http_timeout_sec: int
    sandbox_start_balance: float
    cleanup_keep_days: int
    cleanup_max_logs_mb: int
    log_level: str
    base_dir: str
    logs_dir: str
    state_file: str
    data_dir: str

    @property
    def log_file(self) -> str:
        return os.path.join(self.logs_dir, "bot.log")

    @property
    def max_tickers(self) -> int:
        """Сколько бумаг бот может вести одновременно (сканер/движок/баннер).

        Единая точка истины: и усечение TRADING_TICKERS, и лимит watchlist
        в движке смотрят сюда, так что рассинхрон «константа vs поле»
        исключён (v1.6.0: из-за него падал баннер при старте).
        """
        return MAX_TICKERS


def load_config() -> BotConfig:
    """
    Собирает BotConfig из окружения и валидирует критичные поля.
    Бросает ConfigError, если бот запускать нельзя.
    """
    tinkoff_token = _env_str("TINKOFF_TOKEN")
    sandbox_token = _env_str("TINKOFF_SANDBOX_TOKEN") or tinkoff_token
    groq_key = _env_str("GROQ_API_KEY")
    tg_token = _env_str("TELEGRAM_BOT_TOKEN")
    admin_raw = _env_str("TELEGRAM_ADMIN_ID")

    problems: List[str] = []
    if not sandbox_token and not tinkoff_token:
        problems.append("TINKOFF_SANDBOX_TOKEN / TINKOFF_TOKEN - не задан ни один токен")
    if not groq_key:
        problems.append("GROQ_API_KEY - не задан (нужен для сентимента и аудитора)")
    if not tg_token:
        problems.append("TELEGRAM_BOT_TOKEN - не задан (нужен для управления)")
    if not admin_raw.isdigit():
        problems.append("TELEGRAM_ADMIN_ID - должен быть числовым Telegram ID администратора")
    if problems:
        raise ConfigError("Ошибки конфигурации:\n  - " + "\n  - ".join(problems))

    # Тикеры: верхний регистр, без дублей, максимум MAX_TICKERS.
    tickers = _env_list("TRADING_TICKERS", ("SBER", "GAZP", "LKOH"))
    if len(tickers) > MAX_TICKERS:
        # Логгер ещё не настроен - предупреждаем через stderr.
        print(
            f"[config] ВНИМАНИЕ: TRADING_TICKERS содержит {len(tickers)} тикеров, "
            f"оставляю первые {MAX_TICKERS}: {', '.join(tickers[:MAX_TICKERS])}",
            flush=True,
        )
        tickers = tickers[:MAX_TICKERS]

    logs_dir = os.path.join(BASE_DIR, "logs")
    return BotConfig(
        tinkoff_token=tinkoff_token,
        tinkoff_sandbox_token=sandbox_token,
        groq_api_key=groq_key,
        telegram_bot_token=tg_token,
        telegram_admin_id=int(admin_raw),
        use_sandbox=_env_bool("USE_SANDBOX", True),
        trading_enabled_default=_env_bool("TRADING_ENABLED", False),
        auditor_enabled_default=_env_bool("AUDITOR_ENABLED", True),
        trading_tickers=tickers,
        sma_period=_env_int("SMA_PERIOD", 24, low=2, high=500),
        poll_interval_sec=_env_int("POLL_INTERVAL_SEC", 300, low=30, high=3600),
        max_position_pct=_env_float("MAX_POSITION_PCT", 10.0, low=0.5, high=100.0),
        stop_loss_pct=_env_float("STOP_LOSS_PCT", 3.0, low=0.1, high=50.0),
        take_profit_pct=_env_float("TAKE_PROFIT_PCT", 5.0, low=0.1, high=100.0),
        sentiment_threshold=_env_float("SENTIMENT_THRESHOLD", -0.3, low=-1.0, high=1.0),
        groq_model=_env_str("GROQ_MODEL", "llama-3.1-8b-instant"),
        audit_interval_sec=_env_int("AUDIT_INTERVAL_SEC", 3600, low=300, high=86400),
        audit_log_lines=_env_int("AUDIT_LOG_LINES", 50, low=10, high=200),
        sentiment_ttl_sec=_env_int("SENTIMENT_TTL_SEC", 21600, low=60, high=86400),
        llm_decisions_enabled=_env_bool("LLM_DECISIONS", True),
        trade_cooldown_min=_env_int("TRADE_COOLDOWN_MIN", 120, low=0, high=1440),
        max_buys_per_day=_env_int("MAX_BUYS_PER_DAY", 25, low=1, high=100),
        learning_enabled=_env_bool("LEARNING", True),
        memory_max_trades=_env_int("MEMORY_MAX_TRADES", 400, low=20, high=5000),
        daily_loss_limit_pct=_env_float("DAILY_LOSS_LIMIT_PCT", 2.0,
                                        low=0.0, high=100.0),
        loss_streak_pause_h=_env_float("LOSS_STREAK_PAUSE_H", 4.0,
                                       low=0.0, high=168.0),
        sl_confirm_cycles=_env_int("SL_CONFIRM_CYCLES", 2, low=1, high=10),
        sl_veto_confidence=_env_float("SL_VETO_CONFIDENCE", 0.60,
                                      low=0.0, high=1.0),
        hard_sl_mult=_env_float("HARD_SL_MULT", 1.5, low=1.0, high=5.0),
        trailing_activate_pct=_env_float("TRAILING_ACTIVATE_PCT", 2.0,
                                         low=0.0, high=100.0),
        trailing_drop_pct=_env_float("TRAILING_DROP_PCT", 1.5,
                                     low=0.1, high=100.0),
        llm_early_sell_min_profit_pct=_env_float("LLM_EARLY_SELL_MIN_PROFIT", 0.5,
                                                 low=0.0, high=100.0),
        harvest_tiers=parse_harvest_tiers(
            _env_str("HARVEST_TIERS", "3.0:0.7,5.0:0.4,8.0:0.25")),
        scan_enabled=_env_bool("SCAN_ENABLED", True),
        scan_interval_min=_env_int("SCAN_INTERVAL_MIN", 10, low=5, high=1440),
        scan_min_turnover=_env_float("SCAN_MIN_TURNOVER_MLN", 50.0,
                                     low=0.0, high=1e12),
        scan_min_price=_env_float("SCAN_MIN_PRICE", 5.0, low=0.0, high=1e9),
        scan_deep_n=_env_int("SCAN_DEEP_N", 20, low=5, high=60),
        scan_report_top=_env_int("SCAN_REPORT_TOP", 6, low=3, high=10),
        scan_llm=_env_bool("SCAN_LLM", True),
        rate_limit_rpm=_env_int("RATE_LIMIT_RPM", 90, low=5, high=250),
        news_rss_urls=_env_urls(
            "NEWS_RSS_URLS",
            ("https://www.interfax.ru/rss.asp",),
        ),
        http_timeout_sec=_env_int("HTTP_TIMEOUT_SEC", 15, low=5, high=120),
        sandbox_start_balance=_env_float("SANDBOX_START_BALANCE", 100_000.0, low=1000.0),
        cleanup_keep_days=_env_int("CLEANUP_KEEP_DAYS", 14, low=1, high=365),
        cleanup_max_logs_mb=_env_int("CLEANUP_MAX_LOGS_MB", 20, low=5, high=1024),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        base_dir=BASE_DIR,
        logs_dir=logs_dir,
        state_file=os.path.join(BASE_DIR, "state.json"),
        data_dir=os.path.join(BASE_DIR, "data"),
    )


# ---------------------------------------------------------------------------
# Персистентное runtime-состояние (state.json).
# ---------------------------------------------------------------------------
class StateStore:
    """
    Крошечное JSON-хранилище runtime-состояния.

    Хранит: флаги торговли/аудитора, текущий режим (песочница/реал),
    открытые позиции, последние действия, результат последнего аудита.
    Запись атомарная (tmp + os.replace), файл < 5 КБ - для 15 ГБ диска это ничто.
    """

    _DEFAULTS: Dict[str, Any] = {
        "trading_enabled": False,
        "auditor_enabled": True,
        "use_sandbox": True,
        "positions": {},          # ticker -> {lots, entry_price, entry_ts}
        "last_actions": [],       # до 3 последних строк для /status
        "last_audit_result": "",
        "last_audit_ts": 0,
        "sandbox_confirm_ts": 0,  # время выдачи подтверждения перехода в реал
        "price_history": {},      # ticker -> [[ts, price], ...] для /stats
        "aggressiveness": 50,     # 0..100 - решительность торговли (/set_aggressiveness)
        "activity": 50,           # 0..100 - масштаб времени стратегии (/set_activity)
        "max_invested_pct": 30,   # потолок доли ВСЕХ денег в акциях (/set_invested)
        "decision_log": [],       # последние решения трейдера (память для LLM)
        "last_buy_ts": {},        # ticker -> unix-ts последней покупки (кулдаун)
        "daily_buys": {"date": "", "count": 0},  # покупки за сегодня
        "watchlist": [],          # тикеры под управлением бота (до 6)
    }

    def __init__(self, path: str, initial_overrides: Optional[Dict[str, Any]] = None):
        self._path = path
        self._data: Dict[str, Any] = dict(self._DEFAULTS)
        self._load(initial_overrides)

    # --- внутреннее ---------------------------------------------------------
    def _load(self, overrides: Optional[Dict[str, Any]]) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                if isinstance(stored, dict):
                    self._data.update(stored)
                log.info("Состояние загружено: %s", self._path)
            except (OSError, json.JSONDecodeError) as exc:
                # Битый state.json - не повод не торговать: начинаем с дефолтов.
                log.warning("Не удалось прочитать %s (%s); начинаю с чистого состояния",
                            self._path, exc)
        else:
            # Первый запуск: применяем стартовые значения из окружения.
            if overrides:
                self._data.update(overrides)
            self.save()
            log.info("Создан новый файл состояния: %s", self._path)

    def save(self) -> None:
        """Атомарная запись состояния на диск."""
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=1)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            log.error("Не удалось сохранить состояние в %s: %s", self._path, exc)

    # --- флаги-свойства -------------------------------------------------------
    @property
    def trading_enabled(self) -> bool:
        return bool(self._data["trading_enabled"])

    def set_trading_enabled(self, value: bool) -> None:
        self._data["trading_enabled"] = bool(value)
        self.save()

    @property
    def auditor_enabled(self) -> bool:
        return bool(self._data["auditor_enabled"])

    def set_auditor_enabled(self, value: bool) -> None:
        self._data["auditor_enabled"] = bool(value)
        self.save()

    @property
    def use_sandbox(self) -> bool:
        return bool(self._data["use_sandbox"])

    def set_use_sandbox(self, value: bool) -> None:
        self._data["use_sandbox"] = bool(value)
        self.save()

    # --- позиции ---------------------------------------------------------------
    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        return self._data["positions"]

    def set_position(self, ticker: str, info: Dict[str, Any]) -> None:
        self._data["positions"][ticker] = info
        self.save()

    def drop_position(self, ticker: str) -> None:
        self._data["positions"].pop(ticker, None)
        self.save()

    # --- журнал действий и аудит -----------------------------------------------
    def record_action(self, text: str) -> None:
        """Помнит последние 3 действия для команды /status (deque не нужен)."""
        actions: List[str] = self._data["last_actions"]
        actions.append(f"{time.strftime('%d.%m %H:%M')}: {text}")
        del actions[:-3]
        self.save()

    @property
    def last_actions(self) -> List[str]:
        return list(self._data["last_actions"])

    @property
    def last_audit_result(self) -> str:
        return str(self._data.get("last_audit_result", ""))

    def set_audit_result(self, text: str) -> None:
        self._data["last_audit_result"] = text[:500]
        self._data["last_audit_ts"] = int(time.time())
        self.save()

    @property
    def sandbox_confirm_ts(self) -> float:
        return float(self._data.get("sandbox_confirm_ts", 0))

    def set_sandbox_confirm_ts(self, ts: float) -> None:
        self._data["sandbox_confirm_ts"] = ts
        self.save()

    # --- история цен для /stats -------------------------------------------------
    #: 2 точки/час * 50 часов - файл state.json останется крошечным (< 40 КБ).
    PRICE_HISTORY_MAX_POINTS = 100

    def record_price(self, ticker: str, price: float) -> None:
        """Точка истории цен: движок вызывает её раз в цикл опроса (5 мин)."""
        hist: List[List[float]] = (
            self._data.setdefault("price_history", {}).setdefault(ticker, []))
        hist.append([round(time.time(), 1), round(price, 4)])
        del hist[:-self.PRICE_HISTORY_MAX_POINTS]
        self.save()

    def price_history(self, ticker: str, hours: float = 6.0) -> List[Tuple[float, float]]:
        """Точки (ts, price) за последние `hours` часов, старые - обрезаются."""
        cutoff = time.time() - max(0.5, hours) * 3600.0
        raw = self._data.get("price_history", {}).get(ticker, [])
        return [(float(ts), float(p)) for ts, p in raw if float(ts) >= cutoff]

    # --- агрессивность торговли -------------------------------------------------
    def aggressiveness(self) -> int:
        """Активность торговли 0 (осторожно) .. 100 (агрессивно)."""
        try:
            return int(self._data.get("aggressiveness", 50))
        except (TypeError, ValueError):
            return 50

    def set_aggressiveness(self, value: int) -> None:
        self._data["aggressiveness"] = max(0, min(100, int(value)))
        self.save()

    # --- активность стратегии (масштаб времени) -----------------------------------
    def activity(self) -> int:
        """Масштаб времени стратегии 0 (сутки) .. 100 (часовые движения).

        НЕ то же самое, что агрессивность (пороги/уверенность): активность
        задаёт, за какой период считается рабочая средняя, на каких свечах
        работает движок и какой отрезок графика видит ИИ-трейдер.
        """
        try:
            return int(self._data.get("activity", 50))
        except (TypeError, ValueError):
            return 50

    def set_activity(self, value: int) -> None:
        self._data["activity"] = max(0, min(100, int(value)))
        self.save()

    # --- потолок общей вложенности ------------------------------------------------
    def max_invested_pct(self) -> int:
        """Сколько процентов ВСЕХ денег (кэш + позиции) максимум может
        одновременно находиться в акциях."""
        try:
            return int(self._data.get("max_invested_pct", 30))
        except (TypeError, ValueError):
            return 30

    def set_max_invested_pct(self, value: int) -> None:
        self._data["max_invested_pct"] = max(5, min(100, int(value)))
        self.save()

    # --- журнал решений трейдера (память для LLM) --------------------------------
    def append_decision(self, decision: Dict[str, Any]) -> None:
        """Помнит последние 12 решений: {ts, ticker, action, confidence, price, reason}."""
        log_: List[Dict[str, Any]] = self._data.setdefault("decision_log", [])
        log_.append(decision)
        del log_[:-12]
        self.save()

    def decision_log(self) -> List[Dict[str, Any]]:
        return list(self._data.get("decision_log", []))

    # --- список тикеров под управлением (/watch, /unwatch) ---------------------
    def watchlist(self) -> List[str]:
        """Сохранённый список тикеров (пустой = использовать TRADING_TICKERS)."""
        try:
            raw = self._data.get("watchlist") or []
        except (TypeError, ValueError):
            return []
        return [str(t).strip().upper() for t in raw if str(t).strip()]

    def set_watchlist(self, items: List[str]) -> None:
        """Сохраняет список тикеров (без дублей, верхний регистр)."""
        clean = [str(t).strip().upper() for t in items if str(t).strip()]
        self._data["watchlist"] = list(dict.fromkeys(clean))
        self.save()

    # --- кулдаун и дневной лимит покупок ------------------------------------------
    def last_buy_ts(self, ticker: str) -> float:
        return float(self._data.get("last_buy_ts", {}).get(ticker, 0))

    def set_last_buy_ts(self, ticker: str, ts: float) -> None:
        self._data.setdefault("last_buy_ts", {})[ticker] = ts
        self.save()

    def buys_today(self) -> int:
        """Сколько покупок уже сделано сегодня (сброс в полночь)."""
        today = time.strftime("%Y-%m-%d")
        record = self._data.get("daily_buys", {})
        if record.get("date") != today:
            return 0
        return int(record.get("count", 0))

    def record_buy_today(self) -> None:
        today = time.strftime("%Y-%m-%d")
        record = self._data.get("daily_buys", {})
        if record.get("date") != today:
            self._data["daily_buys"] = {"date": today, "count": 1}
        else:
            record["count"] = int(record.get("count", 0)) + 1
        self.save()
