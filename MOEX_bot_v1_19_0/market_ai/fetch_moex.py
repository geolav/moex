from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
DEFAULT_TICKERS = (
    "SBER", "GAZP", "LKOH", "ROSN", "VTBR", "T", "YDEX", "MOEX",
    "GMKN", "NVTK", "AFLT", "FLOT", "PHOR", "SMLT", "POSI", "VKCO",
    "DOMRF", "MTSS", "CHMF", "NLMK",
)


def fetch_ticker(
    session: requests.Session,
    ticker: str,
    date_from: str,
    date_till: str,
    *,
    interval: int = 1,
    page_size: int = 500,
    pause_sec: float = 0.12,
) -> pd.DataFrame:
    url = BASE_URL.format(ticker=quote(ticker.upper(), safe=""))
    start = 0
    rows: list[dict[str, Any]] = []
    seen_pages: set[tuple[Any, ...]] = set()
    expected_total: int | None = None
    while True:
        params = {
            "from": date_from,
            "till": date_till,
            "interval": int(interval),
            "start": start,
            "limit": page_size,
            "iss.meta": "off",
        }
        response = session.get(url, params=params, timeout=(10, 40))
        response.raise_for_status()
        payload = response.json()
        table = payload.get("candles") or {}
        columns = table.get("columns") or []
        page = table.get("data") or []
        if not page:
            break
        signature = tuple(page[0]) + tuple(page[-1])
        if signature in seen_pages:
            raise RuntimeError(f"MOEX ISS вернул повторяющуюся страницу для {ticker}, start={start}")
        seen_pages.add(signature)
        rows.extend(dict(zip(columns, row)) for row in page)
        cursor = payload.get("candles.cursor") or {}
        ccols = cursor.get("columns") or []
        cdata = cursor.get("data") or []
        if cdata and ccols:
            cursor_row = dict(zip(ccols, cdata[0]))
            try:
                expected_total = int(cursor_row.get("TOTAL"))
            except (TypeError, ValueError):
                pass
        start += len(page)
        if expected_total is not None:
            if start >= expected_total:
                break
        elif len(page) < page_size:
            break
        time.sleep(pause_sec)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["ticker"] = ticker.upper()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка минутных свечей MOEX ISS")
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--till", dest="date_till", required=True, help="YYYY-MM-DD")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--interval", type=int, default=1, choices=(1, 10, 60),
                        help="1=минута, 10=10 минут, 60=час; горизонты модели рассчитаны на interval=1")
    parser.add_argument("--out", default="data/market_ai/moex_1m.csv")
    parser.add_argument("--pause", type=float, default=0.12)
    args = parser.parse_args()

    tickers = list(dict.fromkeys(t.strip().upper() for t in args.tickers.split(",") if t.strip()))
    if not tickers:
        parser.error("Укажите хотя бы один тикер")
    if args.date_from > args.date_till:
        parser.error("--from должен быть не позже --till")

    session = requests.Session()
    session.headers.update({"User-Agent": "MOEX-Market-AI/0.1 (historical research; contact: local user)"})
    frames = []
    errors = []
    for idx, ticker in enumerate(tickers, 1):
        try:
            frame = fetch_ticker(session, ticker, args.date_from, args.date_till,
                                 interval=args.interval, pause_sec=args.pause)
            if frame.empty:
                print(f"[{idx}/{len(tickers)}] {ticker}: нет свечей в выбранном периоде")
                continue
            print(f"[{idx}/{len(tickers)}] {ticker}: {len(frame):,} свечей")
            frames.append(frame)
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
            errors.append((ticker, str(exc)))
            print(f"[{idx}/{len(tickers)}] {ticker}: ОШИБКА: {exc}", file=sys.stderr)
        time.sleep(args.pause)
    if not frames:
        print("Не удалось получить ни одной свечи. Проверьте доступ к iss.moex.com и границы доступной истории.", file=sys.stderr)
        return 2
    result = pd.concat(frames, ignore_index=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\nСохранено: {output} ({len(result):,} свечей, {result['ticker'].nunique()} тикеров)")
    print(f"Период запроса: {args.date_from} — {args.date_till}; интервал={args.interval}")
    if errors:
        print(f"Тикеров с ошибками: {len(errors)}; см. список ниже")
        for ticker, error in errors:
            print(f"  {ticker}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
