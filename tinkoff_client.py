"""
tinkoff_client.py - обёртка над официальным T-Invest API (Т-Инвестиции).

Что умеет:
  - режим ПЕСОЧНИЦА / РЕАЛ переключается на лету (AsyncSandboxClient/AsyncClient);
  - get_price(ticker)      - последняя цена;
  - get_sma(ticker)        - SMA по часовым свечам (для стратегии);
  - get_news(ticker)       - бесплатные RSS-новости по тикеру (в T-Invest API
                             нет новостного эндпоинта, поэтому RSS без ключей);
  - buy / sell             - маркет-ордера (в песочнице - post_sandbox_order);
  - get_portfolio_summary  - кэш, свободные средства, позиции;
  - switch_mode            - безопасное переключение песочница <-> реал.

Надежность:
  - каждый вызов API идёт через _call(): локальный rate-limiter + retry с
    экспоненциальной задержкой; RESOURCE_EXHAUSTED ждёт 45 сек;
  - все ошибки логируются, наружу летит только TinkoffClientError.

Память: никакой истории свечей не храним (deque ограничен кэш-лимитами),
без pandas/numpy. SDK - официальный, после ребрендинга Т-Банка называется
t-tech-investments (импорт из t_tech.invest; старый пакет tinkoff-investments
снят с PyPI).
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import uuid
from collections import deque
from datetime import timedelta
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import aiohttp

# ---------------------------------------------------------------------------
# Импорт официального SDK. Если пакет не установлен, модуль всё равно
# импортируется (нужно для dry-run теста) - start() объяснит, что делать.
# ---------------------------------------------------------------------------
try:
    from t_tech.invest import (
        AsyncClient,
        CandleInterval,
        InstrumentStatus,
        MoneyValue,
        OrderDirection,
        OrderType,
        SecurityTradingStatus,
    )
    from t_tech.invest.exceptions import InvestError
    from t_tech.invest.sandbox.async_client import AsyncSandboxClient
    from t_tech.invest.utils import (
        candle_interval_to_timedelta,
        decimal_to_quotation,
        now as sdk_now,
        quotation_to_decimal,
    )

    SDK_AVAILABLE = True
    _SDK_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - путь только для dry-run
    SDK_AVAILABLE = False
    _SDK_IMPORT_ERROR = exc
    # Заглушки, чтобы интерпретатор не споткнулся об имена при определении класса.
    AsyncClient = None  # type: ignore[assignment]
    AsyncSandboxClient = None  # type: ignore[assignment]
    CandleInterval = None  # type: ignore[assignment]
    InstrumentStatus = None  # type: ignore[assignment]
    MoneyValue = None  # type: ignore[assignment]
    OrderDirection = None  # type: ignore[assignment]
    OrderType = None  # type: ignore[assignment]
    SecurityTradingStatus = None  # type: ignore[assignment]
    InvestError = Exception  # type: ignore[assignment,misc]
    candle_interval_to_timedelta = lambda interval: timedelta(hours=1)  # type: ignore  # noqa: E731
    decimal_to_quotation = lambda value: None  # type: ignore  # noqa: E731
    sdk_now = None  # type: ignore[assignment]
    quotation_to_decimal = lambda value: Decimal(0)  # type: ignore  # noqa: E731

# ---------------------------------------------------------------------------
# Отключение телеметрии SDK (T-Bank error-hub / Sentry).
# SDK при входе в клиент жёстко инициализирует sentry_sdk на захардкоженный
# DSN error-hub.tbank.ru (переменной окружения для отключения нет) и шлёт
# туда ошибки gRPC. TLS до этого хоста с VPS не проверяется, и в лог сыпятся
# SSLCertVerificationError на /api/0/envelope/. На торговлю это не влияет -
# просто глушим отправку, подменяя init-функции на пустые.
# ---------------------------------------------------------------------------
if SDK_AVAILABLE:
    import t_tech.invest.clients as _tt_clients

    def _no_error_hub(_client: Any) -> None:  # вместо init_error_hub
        return None

    async def _no_async_error_hub(_client: Any) -> None:  # вместо async_init_error_hub
        return None

    _tt_clients.init_error_hub = _no_error_hub
    _tt_clients.async_init_error_hub = _no_async_error_hub

log = logging.getLogger("tinkoff")

# Коды gRPC, при которых повтор запроса имеет смысл.
# 2=UNKNOWN, 4=DEADLINE_EXCEEDED, 8=RESOURCE_EXHAUSTED, 13=INTERNAL, 14=UNAVAILABLE
_RETRYABLE_GRPC_CODES = {2, 4, 8, 13, 14}
# Коды, при которых повторять бессмысленно (неверный токен, нет доступа и т.п.).
_FATAL_GRPC_CODES = {3, 5, 7, 16}
# Код RESOURCE_EXHAUSTED: упираемся в лимит API - ждём подольше.
_CODE_RESOURCE_EXHAUSTED = 8

_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"\s+")


def candle_interval_for_step(step_h: float):
    """
    Шаг свечей (в часах) -> (CandleInterval, timedelta).

    Поддерживаем три рабочих масштаба: часовые свечи (1.0),
    15-минутные (0.25) и 5-минутные (1/12). Любое другое значение
    (или SDK без перечисления CandleInterval) -> часовые свечи.
    """
    if not SDK_AVAILABLE or CandleInterval is None:
        return None, timedelta(hours=1.0)
    step = float(step_h)
    if abs(step - 1.0) < 0.01:
        interval = CandleInterval.CANDLE_INTERVAL_HOUR
    elif abs(step - 0.25) < 0.01:
        interval = CandleInterval.CANDLE_INTERVAL_15_MIN
    elif abs(step - 1.0 / 12.0) < 0.01:
        interval = CandleInterval.CANDLE_INTERVAL_5_MIN
    else:
        interval = CandleInterval.CANDLE_INTERVAL_HOUR
    return interval, candle_interval_to_timedelta(interval)


class TinkoffClientError(RuntimeError):
    """Невосполнимая ошибка работы с T-Invest API."""


class _RateLimiter:
    """
    Простейший скользящий лимитер: не более rpm запросов за 60 секунд.

    Официальные лимиты T-Invest API: 100-300 запросов/мин на метод
    (getCandles/getLastPrices/postOrder 300, getPortfolio 200, getAccounts 100).
    Дефолт 90 rpm на ВСЕ вызовы суммарно - многократный запас.
    """

    __slots__ = ("_rpm", "_window")

    def __init__(self, rpm: int) -> None:
        self._rpm = max(1, int(rpm))
        self._window: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            now_mono = time.monotonic()
            while self._window and now_mono - self._window[0] > 60.0:
                self._window.popleft()
            if len(self._window) < self._rpm:
                self._window.append(now_mono)
                return
            sleep_sec = 60.0 - (now_mono - self._window[0]) + 0.05
            await asyncio.sleep(min(max(sleep_sec, 0.05), 60.0))


class TinkoffClient:
    """Асинхронная обёртка над T-Invest API с переключением контуров."""

    def __init__(self, config, state) -> None:
        self._cfg = config
        self._state = state
        self._cm: Any = None            # AsyncClient/AsyncSandboxClient (context manager)
        self._api: Any = None           # живой набор сервисов (services)
        self._account_id: Optional[str] = None
        self._instruments: Dict[str, Dict[str, Any]] = {}   # ticker -> {figi, lot, name}
        self._ticker_by_figi: Dict[str, str] = {}
        # Полная вселенная акций основного режима торгов TQBR (~250 бумаг):
        # нужна сканеру и ручным сделкам (/buy любой бумаги без рестарта).
        self._universe: Dict[str, Dict[str, Any]] = {}
        self._limiter = _RateLimiter(config.rate_limit_rpm)
        self._switch_lock = asyncio.Lock()
        self._http: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------ props
    @property
    def use_sandbox(self) -> bool:
        return self._state.use_sandbox

    @property
    def account_id(self) -> str:
        return self._account_id or ""

    @property
    def connected(self) -> bool:
        return self._api is not None

    # ----------------------------------------------------------- жизненный цикл
    async def start(self) -> None:
        """Подключается к API в текущем режиме (песочница по умолчанию)."""
        if not SDK_AVAILABLE:
            raise TinkoffClientError(
                "SDK Т-Инвестиций не установлен. Выполните: pip install "
                "--extra-index-url https://opensource.tbank.ru/api/v4/projects/238/"
                "packages/pypi/simple t-tech-investments"
                f" (исходная ошибка импорта: {_SDK_IMPORT_ERROR})"
            )
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._cfg.http_timeout_sec)
        )
        await self._connect()

    async def close(self) -> None:
        async with self._switch_lock:
            await self._close_api()
            if self._http is not None:
                await self._http.close()
                self._http = None

    async def switch_mode(self, use_sandbox: bool) -> None:
        """Переключение песочница <-> реальный счёт на живом процессе."""
        async with self._switch_lock:
            if self._api is not None and use_sandbox == self._state.use_sandbox:
                return
            await self._close_api()
            self._state.set_use_sandbox(use_sandbox)
            await self._connect()

    async def _close_api(self) -> None:
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - закрытие не должно ронять бот
                log.warning("Ошибка при закрытии gRPC-канала: %s", exc)
        self._cm = None
        self._api = None
        self._account_id = None
        self._instruments.clear()
        self._ticker_by_figi.clear()
        self._universe.clear()

    async def _connect(self) -> None:
        use_sandbox = self._state.use_sandbox
        token = self._cfg.tinkoff_sandbox_token if use_sandbox else self._cfg.tinkoff_token
        if not token:
            raise TinkoffClientError(
                "Нет токена для выбранного режима: заполните "
                + ("TINKOFF_SANDBOX_TOKEN" if use_sandbox else "TINKOFF_TOKEN")
                + " в .env"
            )
        client_cls = AsyncSandboxClient if use_sandbox else AsyncClient
        log.info("Подключаюсь к T-Invest API: режим %s",
                 "ПЕСОЧНИЦА" if use_sandbox else "РЕАЛЬНЫЙ СЧЁТ")
        self._cm = client_cls(token, app_name="moex-trading-bot")
        self._api = await self._cm.__aenter__()
        self._account_id = await self._resolve_account(use_sandbox)
        await self._load_instruments()
        log.info("Аккаунт %s готов, инструментов загружено: %d",
                 self._mask(self._account_id), len(self._instruments))

    @staticmethod
    def _mask(value: Optional[str]) -> str:
        return (value or "")[:8] + "..." if value else "?"

    async def _resolve_account(self, use_sandbox: bool) -> str:
        """Находит (или создаёт в песочнице) торговый счёт."""
        if use_sandbox:
            try:
                response = await self._call(
                    "get_sandbox_accounts", lambda: self._api.sandbox.get_sandbox_accounts()
                )
                accounts = response.accounts
            except TinkoffClientError:
                accounts = []
            if not accounts:
                opened = await self._call(
                    "open_sandbox_account", lambda: self._api.sandbox.open_sandbox_account()
                )
                account_id = opened.account_id
                log.info("Создан новый счёт песочницы %s, зачисляю стартовый капитал...",
                         self._mask(account_id))
                await self._sandbox_pay_in(account_id, self._cfg.sandbox_start_balance)
                return account_id
            return accounts[0].id

        response = await self._call("get_accounts", lambda: self._api.users.get_accounts())
        if not response.accounts:
            raise TinkoffClientError("На реальном счёте не найдено ни одного аккаунта")
        # Предпочитаем брокерский счёт (type=1 ACCOUNT_TYPE_TINKOFF) перед ИИС.
        for account in response.accounts:
            if int(getattr(account, "type", 0)) == 1:
                return account.id
        return response.accounts[0].id

    async def _sandbox_pay_in(self, account_id: str, amount_rub: float) -> None:
        """Разовое зачисление виртуальных денег на счёт песочницы."""
        quotation = decimal_to_quotation(Decimal(str(amount_rub)))
        await self._call(
            "sandbox_pay_in",
            lambda: self._api.sandbox.sandbox_pay_in(
                account_id=account_id,
                amount=MoneyValue(units=quotation.units, nano=quotation.nano, currency="rub"),
            ),
        )
        log.info("Песочница пополнена на %.2f руб.", amount_rub)

    async def _load_instruments(self) -> None:
        """Кэш figi/лот/название: полный список TQBR + рабочие тикеры конфига."""
        response = await self._call(
            "instruments.shares",
            lambda: self._api.instruments.shares(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            ),
        )
        wanted = set(self._cfg.trading_tickers)
        universe: Dict[str, Dict[str, Any]] = {}
        found: Dict[str, Dict[str, Any]] = {}
        for share in response.instruments:
            info = {
                "figi": share.figi,
                "lot": int(share.lot) or 1,
                "name": (share.name or "").strip(),
                "class_code": share.class_code,
            }
            # Основная площадка МосБиржи - TQBR; прочие классы храним в резерве.
            if share.class_code == "TQBR" or share.ticker not in universe:
                universe[share.ticker] = info
            if share.ticker in wanted and (
                    share.class_code == "TQBR" or share.ticker not in found):
                found[share.ticker] = info
        # Предпочитаем TQBR: перезаписываем найденное по другим класс-кодам.
        for share in response.instruments:
            if share.class_code == "TQBR" and share.ticker in found:
                found[share.ticker] = {
                    "figi": share.figi,
                    "lot": int(share.lot) or 1,
                    "name": (share.name or "").strip(),
                    "class_code": share.class_code,
                }
        self._universe = universe
        self._instruments = found
        self._ticker_by_figi = {info["figi"]: t for t, info in found.items()}
        missing = wanted - set(found)
        if missing:
            log.warning("Тикеры не найдены на MOEX (проверьте написание): %s",
                        ", ".join(sorted(missing)))
        log.info("Вселенная бумаг TQBR: %d шт (рабочих тикеров: %d)",
                 len(universe), len(found))

    # ------------------------------------------------------- вызов API с retry
    async def _call(
        self,
        what: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        tries: int = 3,
        base_delay: float = 2.0,
    ) -> Any:
        """
        Универсальный вызов gRPC-метода: rate-limit + retry + логирование.

        coro_factory - функция, создающая НОВУЮ корутину на каждую попытку
        (повторно await-ить старую корутину нельзя).
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, tries + 1):
            await self._limiter.acquire()
            try:
                return await coro_factory()
            except InvestError as exc:
                code = getattr(exc, "code", None)
                code_num = 0
                if code is not None:
                    value = getattr(code, "value", None)
                    code_num = value[0] if isinstance(value, tuple) else int(value or 0)
                if code_num in _FATAL_GRPC_CODES:
                    raise TinkoffClientError(
                        f"{what}: фатальная ошибка API (код {code_num}): "
                        f"{getattr(exc, 'details', exc)}"
                    ) from exc
                last_error = exc
                # RESOURCE_EXHAUSTED (лимит API) -> долгая пауза, иначе экспонента.
                delay = 45.0 if code_num == _CODE_RESOURCE_EXHAUSTED else base_delay * 2 ** (attempt - 1)
                log.warning("T-Invest %s: ошибка код %s (попытка %d/%d), повтор через %.0f сек: %s",
                            what, code_num, attempt, tries, delay, getattr(exc, "details", exc))
                await asyncio.sleep(delay)
            except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                delay = base_delay * 2 ** (attempt - 1)
                log.warning("T-Invest %s: сеть/таймаут (попытка %d/%d), повтор через %.1f сек: %s",
                            what, attempt, tries, delay, exc)
                await asyncio.sleep(delay)
        raise TinkoffClientError(f"{what}: не удалось после {tries} попыток: {last_error}")

    # ------------------------------------------------------------------- рынок
    def universe(self) -> Dict[str, Dict[str, Any]]:
        """Все известные бумаги TQBR (тикер -> figi/лот/название)."""
        return dict(self._universe)

    def ensure_instrument(self, ticker: str) -> bool:
        """Подключает тикер к работе (из вселенной TQBR) или сообщает, что нет.

        Нужен для ручных сделок и сканера: бумага может не быть в списке
        конфига, но figi/лот у клиента уже есть - просто активируем кэш.
        """
        ticker = (ticker or "").strip().upper()
        if ticker in self._instruments:
            return True
        info = self._universe.get(ticker)
        if not info:
            return False
        self._instruments[ticker] = info
        self._ticker_by_figi[info["figi"]] = ticker
        log.info("Тикер %s подключён к работе из вселенной TQBR", ticker)
        return True

    #: Открытый API МосБиржи: весь режим TQBR одним запросом, без ключей.
    MOEX_ISS_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares"
                    "/boards/TQBR/securities.json")

    async def get_market_snapshot(self) -> List[Dict[str, Any]]:
        """Снимок всего рынка TQBR одним HTTP-запросом (без ключей и лимитов).

        Возвращает список {ticker, price, day_chg_pct, turnover,
        trading_status, updatetime}:
        turnover - оборот за сегодня в рублях (важно для фильтра неликвида).

        ВАЖНО (v1.7.0): живые поля LAST / LASTTOPREVPRICE / VALTODAY живут
        в блоке MARKETDATA, а НЕ в securities - раньше мы просили их у
        securities, ISS молча выкидывал незнакомые колонки и возвращал
        только SECID+PREVPRICE, из-за чего обороты ВСЕГДА были нулями:
        сканер и днём писал "рынок не торгуется" и никогда не находил
        кандидатов. Теперь запрашиваем оба блока и склеиваем их по SECID.
        Ошибка сети/формата -> TinkoffClientError (сканер покажет пропуск).
        """
        url = (self.MOEX_ISS_URL
               + "?iss.meta=off&iss.only=securities,marketdata"
                 "&securities.columns=SECID,PREVPRICE"
                 "&marketdata.columns=SECID,LAST,LASTTOPREVPRICE,"
                 "VALTODAY,TRADINGSTATUS,UPDATETIME")
        if self._http is None:
            raise TinkoffClientError("HTTP-сессия ещё не открыта (client.start())")
        try:
            async with self._http.get(url) as response:
                if response.status != 200:
                    raise TinkoffClientError(f"MOEX ISS: HTTP {response.status}")
                payload = await response.json(content_type=None)
        except TinkoffClientError:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
            raise TinkoffClientError(f"MOEX ISS недоступен: {exc}") from exc

        def _block(name: str) -> Tuple[Dict[str, int], List[list]]:
            block = payload.get(name, {}) if isinstance(payload, dict) else {}
            cols = block.get("columns", []) or []
            return ({name: i for i, name in enumerate(cols)},
                    block.get("data", []) or [])

        sec_idx, sec_rows = _block("securities")
        md_idx, md_rows = _block("marketdata")
        if "SECID" not in sec_idx and "SECID" not in md_idx:
            raise TinkoffClientError("MOEX ISS: неожиданный формат ответа")

        def _num(row: list, idx: Dict[str, int], name: str) -> float:
            if name not in idx:
                return 0.0
            value = row[idx[name]] if idx[name] < len(row) else None
            try:
                return float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _txt(row: list, idx: Dict[str, int], name: str) -> str:
            if name not in idx or idx[name] >= len(row):
                return ""
            value = row[idx[name]]
            return str(value) if value is not None else ""

        # Склейка: marketdata - живые цифры, securities - PREVPRICE на случай,
        # если LAST по бумаге пуст (торгов по ней сегодня ещё не было).
        prev_price: Dict[str, float] = {}
        for row in sec_rows:
            if sec_idx.get("SECID", 0) < len(row) and row[sec_idx["SECID"]]:
                prev_price[str(row[sec_idx["SECID"]])] = _num(row, sec_idx, "PREVPRICE")
        out: List[Dict[str, Any]] = []
        for row in md_rows:
            if md_idx.get("SECID", 0) >= len(row) or not row[md_idx["SECID"]]:
                continue
            secid = str(row[md_idx["SECID"]])
            price = _num(row, md_idx, "LAST") or prev_price.get(secid, 0.0)
            out.append({
                "ticker": secid,
                "price": price,
                "day_chg_pct": _num(row, md_idx, "LASTTOPREVPRICE"),
                "turnover": _num(row, md_idx, "VALTODAY"),
                "trading_status": _txt(row, md_idx, "TRADINGSTATUS"),
                "updatetime": _txt(row, md_idx, "UPDATETIME"),
            })
        # Бумаги, которых нет в marketdata (редкий случай), добавляем из
        # securities с нулевым оборотом - чтобы не терять вселенную.
        seen = {r["ticker"] for r in out}
        for secid, prev in prev_price.items():
            if secid not in seen:
                out.append({"ticker": secid, "price": prev,
                            "day_chg_pct": 0.0, "turnover": 0.0,
                            "trading_status": "", "updatetime": ""})
        log.info("Снимок MOEX ISS: %d бумаг (marketdata: %d, securities: %d)",
                 len(out), len(md_rows), len(sec_rows))
        return out

    def get_lot_size(self, ticker: str) -> int:
        """Размер лота бумаги из кэша инструментов (1, если тикер неизвестен)."""
        return int(self._instruments.get(ticker, {}).get("lot", 1))

    def instrument_name(self, ticker: str) -> str:
        """Название компании из кэша инструментов (для сообщений)."""
        return str(self._instruments.get(ticker, {}).get("name", ""))

    async def get_price(self, ticker: str) -> Optional[float]:
        """Последняя цена акции; None, если цены нет."""
        instrument = self._instruments.get(ticker)
        if instrument is None:
            log.warning("get_price: тикер %s не в кэше инструментов", ticker)
            return None
        response = await self._call(
            "get_last_prices",
            lambda: self._api.market_data.get_last_prices(figi=[instrument["figi"]]),
        )
        if not response.last_prices:
            return None
        price = quotation_to_decimal(response.last_prices[0].price)
        return float(price) if price else None

    async def get_spread_pct(self, ticker: str) -> Optional[float]:
        """Текущий спред по стакану в процентах от середины (v1.8.0).

        Нужен движку перед покупкой: спред > MAX_SPREAD_PCT - сделку лучше
        пропустить (издержки съедят ход). None - стакан недоступен/пуст:
        вызывающий код считает проверку непройденной "по умолчанию ок".
        """
        instrument = self._instruments.get(ticker)
        if instrument is None:
            return None
        try:
            response = await self._call(
                "get_order_book",
                lambda: self._api.market_data.get_order_book(
                    figi=instrument["figi"], depth=1),
            )
        except TinkoffClientError as exc:
            log.debug("%s: стакан недоступен (%s)", ticker, exc)
            return None
        bids = list(getattr(response, "bids", []) or [])
        asks = list(getattr(response, "asks", []) or [])
        if not bids or not asks:
            return None
        try:
            bid = float(quotation_to_decimal(bids[0].price))
            ask = float(quotation_to_decimal(asks[0].price))
        except (TypeError, ValueError, AttributeError, IndexError):
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None
        return (ask - bid) / mid * 100.0

    async def is_tradable(self, ticker: str) -> Optional[bool]:
        """Открыта ли сейчас торговая сессия по бумаге (v1.6.1).

        True  - NORMAL_TRADING и API-торговля разрешена: можно ставить ордера;
        False - сессия закрыта/выходной/бумага недоступна. Именно это было
                источником INVALID_ARGUMENT 30079 ночью: официальное значение
                кода - "instrument is not available for trading";
        None  - статус неизвестен (ошибка API/нет SDK): вызывающий код работает
                по-старому (fail-open) - статус не должен остановить торговлю.
        """
        instrument = self._instruments.get(ticker)
        if instrument is None:
            return None
        if SecurityTradingStatus is None:      # SDK не установлен (dry-run)
            return None
        try:
            response = await self._call(
                "get_trading_status",
                lambda: self._api.market_data.get_trading_status(
                    figi=instrument["figi"]),
            )
        except TinkoffClientError as exc:
            log.warning("%s: торговый статус недоступен (%s) - считаю торгуемой",
                        ticker, exc)
            return None
        api_ok = bool(getattr(response, "api_trade_available_flag", True))
        status = getattr(response, "trading_status", None)
        normal = (status == SecurityTradingStatus
                  .SECURITY_TRADING_STATUS_NORMAL_TRADING)
        return bool(api_ok and normal)

    async def get_closes(self, ticker: str, *, count: int) -> List[float]:
        """
        Цены закрытия ПОСЛЕДНИХ закрытых часовых свечей (<= count значений).

        Текущая (незакрытая) свеча отбрасывается: сигнал по неполной свече
        даёт ложные входы.
        """
        instrument = self._instruments.get(ticker)
        if instrument is None:
            return []
        interval = CandleInterval.CANDLE_INTERVAL_HOUR
        interval_td = candle_interval_to_timedelta(interval)
        now_utc = sdk_now()
        response = await self._call(
            "get_candles",
            lambda: self._api.market_data.get_candles(
                figi=instrument["figi"],
                interval=interval,
                from_=now_utc - interval_td * (count + 3),
                to=now_utc,
            ),
        )
        closes: List[float] = []
        for candle in response.candles:
            if candle.time + interval_td <= now_utc:  # свеча гарантированно закрыта
                closes.append(float(quotation_to_decimal(candle.close)))
        return closes[-count:]

    async def get_candles_history(self, ticker: str, hours_back: float,
                                  step_h: float = 1.0) -> List[Tuple[float, float]]:
        """
        Закрытые свечи за последние `hours_back` часов: [(ts_закрытия, цена)].

        step_h - длительность свечи в часах: 1.0 - часовые, 0.25 - 15-минутные,
        1/12 - 5-минутные (неизвестное значение -> часовые; без SDK - тоже
        часовые). ts_закрытия - unix-время закрытия свечи: движок и /stats
        работают с ним, а не с открытием (свеча 10:00-11:00 реально
        описывает цену в 11:00).
        """
        instrument = self._instruments.get(ticker)
        if instrument is None:
            return []

        interval, interval_td = candle_interval_for_step(step_h)
        if interval is None or sdk_now is None:
            # SDK недоступен (dry-run) - данных нет, движок уйдёт в фолбэк.
            return []
        now_utc = sdk_now()
        # Свечей запрашиваем с запасом: торги идут ~14 ч/сутки, а не 24.
        count = int(float(hours_back) / max(step_h, 1e-9) * 1.5) + 3
        count = max(10, min(400, count))
        response = await self._call(
            "get_candles",
            lambda: self._api.market_data.get_candles(
                figi=instrument["figi"],
                interval=interval,
                from_=now_utc - interval_td * (count + 3),
                to=now_utc,
            ),
        )
        pairs: List[Tuple[float, float]] = []
        for candle in response.candles:
            close_dt = candle.time + interval_td
            if close_dt <= now_utc:  # свеча гарантированно закрыта
                pairs.append((float(close_dt.timestamp()),
                              float(quotation_to_decimal(candle.close))))
        pairs.sort(key=lambda p: p[0])
        return pairs

    async def get_hourly_history(self, ticker: str, hours_back: float
                                 ) -> List[Tuple[float, float]]:
        """Часовые закрытые свечи за `hours_back` часов: [(ts_закрытия, цена)]."""
        return await self.get_candles_history(ticker, hours_back, step_h=1.0)

    async def get_sma(self, ticker: str) -> Optional[float]:
        """Простая скользящая средняя за SMA_PERIOD закрытых часовых свечей."""
        closes = await self.get_closes(ticker, count=self._cfg.sma_period)
        if len(closes) < self._cfg.sma_period:
            log.info("get_sma(%s): мало данных (%d из %d свечей)",
                     ticker, len(closes), self._cfg.sma_period)
            return None
        return sum(closes) / len(closes)

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Сводка портфеля: кэш, всего, позиции (тикер, лоты, количество, цена)."""
        response = await self._call(
            "get_portfolio",
            lambda: self._api.operations.get_portfolio(account_id=self._account_id),
        )

        def _to_float(quotation: Any) -> float:
            return float(quotation_to_decimal(quotation)) if quotation is not None else 0.0

        positions: List[Dict[str, Any]] = []
        for position in response.positions:
            quantity = _to_float(position.quantity)
            ticker = self._ticker_by_figi.get(position.figi, position.figi)
            lots = int(getattr(position, "quantity_lots", None) and
                       quotation_to_decimal(position.quantity_lots)) or 0
            if not lots:  # quantity_lots иногда пуст - считаем из лота бумаги
                lot_size = self._instruments.get(ticker, {}).get("lot", 1)
                lots = int(quantity // lot_size)
            positions.append({
                "ticker": ticker,
                "lots": lots,
                "quantity": quantity,
                "price": _to_float(position.current_price),
                "avg_price": _to_float(position.average_position_price),
            })
        return {
            "total": _to_float(response.total_amount_portfolio),
            "cash": _to_float(response.total_amount_currencies),
            "positions": positions,
        }

    # ------------------------------------------------------------------ ордера
    async def _post_order(self, ticker: str, lots: int, direction: Any,
                          price: Optional[float] = None) -> Dict[str, Any]:
        """Рыночный ордер: в песочнице post_sandbox_order, в реале post_order.

        ВАЖНО: для акций MOEX (класс TQBR) брокер требует поле price даже у
        рыночной заявки (иначе INVALID_ARGUMENT 30079) - поэтому цену
        передаём всегда, когда она известна."""
        instrument = self._instruments.get(ticker)
        if instrument is None:
            raise TinkoffClientError(f"Ордер {ticker}: тикер не найден в кэше инструментов")
        # order_id - ключ идемпотентности: при повторе сети брокер не задвоит заявку.
        order_id = str(uuid.uuid4())

        if self.use_sandbox:
            response = await self._call(
                "post_sandbox_order",
                lambda: self._api.sandbox.post_sandbox_order(
                    figi=instrument["figi"],
                    quantity=lots,
                    direction=direction,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    account_id=self._account_id,
                    order_id=order_id,
                    price=decimal_to_quotation(Decimal(str(round(price, 2)))) if price else None,
                ),
            )
        else:
            response = await self._call(
                "post_order",
                lambda: self._api.orders.post_order(
                    figi=instrument["figi"],
                    quantity=lots,
                    direction=direction,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    account_id=self._account_id,
                    order_id=order_id,
                    price=decimal_to_quotation(Decimal(str(round(price, 2)))) if price else None,
                ),
            )
        return {
            "order_id": str(getattr(response, "order_id", order_id)),
            "status": int(getattr(response, "status", 0) or 0),
            "lots_executed": int(getattr(response, "lots_executed", 0) or 0),
            "requested_lots": lots,
        }

    async def buy(self, ticker: str, lots: int, price: float = 0.0) -> Dict[str, Any]:
        return await self._post_order(ticker, lots, OrderDirection.ORDER_DIRECTION_BUY, price)

    async def sell(self, ticker: str, lots: int, price: float = 0.0) -> Dict[str, Any]:
        return await self._post_order(ticker, lots, OrderDirection.ORDER_DIRECTION_SELL, price)

    # ------------------------------------------------------------------ новости
    async def get_news(self, ticker: str, limit: int = 8) -> List[str]:
        """
        Бесплатные RSS-новости по тикеру (поиск по тикеру и названию компании).

        T-Invest API не имеет новостного эндпоинта, поэтому используем
        открытые RSS-ленты из конфига (по умолчанию Интерфакс). Парсинг -
        stdlib xml.etree, без тяжёлых зависимостей. Ленты опрашиваются
        СТРОГО последовательно (требование по памяти, никаких gather).
        """
        if self._http is None:
            return []
        instrument = self._instruments.get(ticker)
        keywords = {ticker.lower()}
        if instrument and instrument.get("name"):
            first_word = instrument["name"].split()[0].lower()
            if len(first_word) >= 4:  # отсекаем предлоги/короткие слова
                keywords.add(first_word)

        found: List[str] = []
        for url in self._cfg.news_rss_urls:
            if len(found) >= limit:
                break
            raw_bytes = await self._fetch_url(url)
            if raw_bytes is None:
                continue
            for item in self._parse_rss_items(raw_bytes):
                text = _SPACES_RE.sub(" ", _TAG_RE.sub(" ", item)).strip()
                lowered = text.lower()
                if any(keyword in lowered for keyword in keywords):
                    found.append(html.unescape(text)[:400])
                if len(found) >= limit:
                    break
        return found

    async def _fetch_url(self, url: str) -> Optional[bytes]:
        try:
            async with self._http.get(url) as response:
                if response.status != 200:
                    log.warning("RSS %s: HTTP %d", url, response.status)
                    return None
                return await response.read()
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
            log.warning("RSS %s недоступна: %s", url, exc)
            return None

    @staticmethod
    def _parse_rss_items(raw_bytes: bytes) -> List[str]:
        """Достаёт 'title. description' из RSS 2.0 / Atom (максимум 30 позиций)."""
        try:
            root = ET.fromstring(raw_bytes)
        except ET.ParseError as exc:
            log.warning("RSS не распарсилась: %s", exc)
            return []
        items: List[str] = []
        for item in root.iter():
            tag = item.tag.rsplit("}", 1)[-1]
            if tag != "item" and tag != "entry":
                continue
            title = description = ""
            for child in item:
                child_tag = child.tag.rsplit("}", 1)[-1]
                if child_tag == "title" and child.text:
                    title = child.text.strip()
                elif child_tag in ("description", "summary") and child.text:
                    description = child.text.strip()
            if title:
                items.append(f"{title}. {description}" if description else title)
            if len(items) >= 30:
                break
        return items
