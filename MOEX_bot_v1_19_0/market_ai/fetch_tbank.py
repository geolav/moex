from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_URL = (
    "https://invest-public-api.tbank.ru/rest/"
    "tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
)
MOSCOW = ZoneInfo("Europe/Moscow")
UTC = timezone.utc
DEFAULT_TICKERS = ("SBER", "GAZP", "LKOH", "ROSN", "VTBR", "T", "YDEX", "MOEX", "AFLT", "FLOT")


def _quotation(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.get("units", 0) or 0) + float(value.get("nano", 0) or 0) / 1_000_000_000.0


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch_day(
    session: requests.Session,
    instrument_id: str,
    day: date,
    *,
    retries: int = 5,
) -> list[dict]:
    # GetCandles для 1m ограничен периодом до одного дня и 2400 свечами.
    start_local = datetime.combine(day, dtime.min, tzinfo=MOSCOW)
    end_local = start_local + timedelta(days=1)
    body = {
        "from": _utc_iso(start_local),
        "to": _utc_iso(end_local),
        "interval": "CANDLE_INTERVAL_1_MIN",
        "instrumentId": instrument_id,
        "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
        "limit": 2400,
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.post(API_URL, json=body, timeout=(10, 45))
            if response.status_code == 429 or response.status_code >= 500:
                wait = min(30.0, 1.5 ** attempt)
                if attempt < retries:
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            payload = response.json()
            candles = payload.get("candles")
            if candles is None:
                raise ValueError(f"В ответе GetCandles нет candles: {str(payload)[:400]}")
            rows = []
            for candle in candles:
                ts = pd.to_datetime(candle.get("time"), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                local_naive = ts.tz_convert(MOSCOW).tz_localize(None)
                open_price = _quotation(candle.get("open"))
                high = _quotation(candle.get("high"))
                low = _quotation(candle.get("low"))
                close = _quotation(candle.get("close"))
                volume = int(candle.get("volume", 0) or 0)
                if min(open_price, high, low, close) <= 0:
                    continue
                rows.append({
                    "ticker": instrument_id.split("_")[0],
                    "instrument_id": instrument_id,
                    "begin": local_naive.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": open_price, "high": high, "low": low, "close": close,
                    "volume": volume,
                    "value": close * volume,
                    "is_complete": candle.get("isComplete", True),
                })
            return rows
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(30.0, 1.5 ** attempt))
    raise RuntimeError(f"{instrument_id} {day}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка минутных свечей Т-Инвестиции (T-Bank API)")
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--till", dest="date_till", required=True, help="YYYY-MM-DD")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--suffix", default="_TQBR", help="класс инструмента, по умолчанию TQBR")
    parser.add_argument("--out", default="data/market_ai/tbank_1m.csv")
    parser.add_argument("--token-env", default="TINKOFF_TOKEN",
                        help="имя env-переменной с API-токеном; токен не пишется в CSV")
    parser.add_argument("--pause", type=float, default=0.60,
                        help="пауза между запросами, чтобы не упираться в rate limit")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="включить незакрытые свечи; по умолчанию они удаляются")
    args = parser.parse_args()

    token = os.getenv(args.token_env, "").strip()
    if not token:
        print(f"Не найдена переменная окружения {args.token_env}. Токен передавайте через env, не аргументом CLI.", file=sys.stderr)
        return 2
    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_till)
    if start > end:
        parser.error("--from должен быть не позже --till")
    tickers = list(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    if not tickers:
        parser.error("Укажите хотя бы один тикер")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MOEX-Market-AI/0.1",
    })
    all_rows = []
    errors = []
    n_days = (end - start).days + 1
    total_jobs = len(tickers) * n_days
    completed = 0
    for ticker in tickers:
        instrument_id = ticker if "_" in ticker else f"{ticker}{args.suffix}"
        current = start
        while current <= end:
            completed += 1
            try:
                rows = fetch_day(session, instrument_id, current)
                if not args.include_incomplete:
                    rows = [row for row in rows if bool(row.get("is_complete", True))]
                all_rows.extend(rows)
                print(f"[{completed}/{total_jobs}] {instrument_id} {current}: {len(rows)} свечей")
            except Exception as exc:
                errors.append((instrument_id, current.isoformat(), str(exc)))
                print(f"[{completed}/{total_jobs}] {instrument_id} {current}: ОШИБКА: {exc}", file=sys.stderr)
            current += timedelta(days=1)
            if args.pause > 0:
                time.sleep(args.pause)

    if not all_rows:
        print("Не получено ни одной свечи. Проверьте токен, FIGI/instrumentId, class code и доступность истории.", file=sys.stderr)
        return 2
    frame = pd.DataFrame(all_rows).drop_duplicates(["ticker", "begin"], keep="last")
    frame = frame.sort_values(["ticker", "begin"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nСохранено: {out}; {len(frame):,} свечей; тикеров={frame.ticker.nunique()}")
    print(f"Период: {start} — {end}; ошибки запросов: {len(errors)}")
    if errors:
        error_file = out.with_suffix(".errors.txt")
        error_file.write_text("\n".join(f"{t} {d}: {e}" for t, d, e in errors), encoding="utf-8")
        print(f"Подробности ошибок: {error_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
