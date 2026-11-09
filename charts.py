"""
charts.py - групповые графики для Telegram: несколько панелей на одной картинке.

Зачем: когда бот ведёт 6 бумаг (или сканер нашёл 6 кандидатов), слать 6
отдельных фото - долго и захламляет чат. Этот модуль собирает до 3 панелей
(цена + средняя + пунктир цены покупки) в ОДНУ картинку: 6 тикеров =
ровно 2 фото. Рисует в памяти (Agg), возвращает PNG-байты - файлы на диск
не пишутся, VPS-диск не растёт.

Зависимость matplotlib - опциональная: если её нет, функции возвращают
пустой список, а вызывающий код отправляет текстовый фолбэк.
"""
from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("charts")

#: Панелей (графиков) на одной картинке по умолчанию: 6 тикеров -> 2 фото.
DEFAULT_PER_PHOTO = 3


def _clean_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Отбрасывает битые точки (None/нечисловые ts или цена, price <= 0).

    v1.9.0: графики не должны падать целиком из-за одной плохой точки,
    случайно попавшей в state.json или пришедшей с биржи.
    """
    clean: List[Tuple[float, float]] = []
    for point in points or []:
        try:
            ts, price = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(ts) and math.isfinite(price) and price > 0:
            clean.append((ts, price))
    return clean


def sma_series(points: Sequence[Tuple[float, float]], period: int
               ) -> List[Tuple[float, float]]:
    """Скользящая средняя по ряду (ts, price): [(ts, sma)] длиной len-period+1."""
    closes = [p for _, p in points]
    out: List[Tuple[float, float]] = []
    for i in range(len(closes)):
        if i + 1 >= period:
            out.append((points[i][0], sum(closes[i + 1 - period:i + 1]) / period))
    return out


def render_group_pngs(series: List[Dict[str, Any]], *,
                      per_photo: int = DEFAULT_PER_PHOTO,
                      suptitle: str = "",
                      positions: Optional[Dict[str, dict]] = None) -> List[bytes]:
    """
    Рисует графики группами и возвращает список PNG (list[bytes]).

    Каждый элемент series:
      {"ticker": str,               - заголовок панели;
       "points": [(ts, price)],     - линия цены (ts = unix-время);
       "sma": [(ts, value)]|None,   - пунктирная линия средней (опционально);
       "subtitle": str}             - приписка к заголовку (например "+2.1%").

    positions: {ticker: {"entry_price": float}} - нарисует красный пунктир
    цены покупки по бумагам с открытой позицией.
    """
    if not series:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - matplotlib может быть не установлен
        log.warning("matplotlib недоступен, графики не строю: %s", exc)
        return []

    per_photo = max(1, int(per_photo))
    photos: List[bytes] = []
    for offset in range(0, len(series), per_photo):
        chunk = series[offset:offset + per_photo]
        # v1.9.0: чистим точки заранее и не рисуем панели без данных.
        clean_chunk = []
        for item in chunk:
            pts = _clean_points(item.get("points") or [])
            if len(pts) < 2:
                continue
            item = dict(item)
            item["points"] = pts
            sma = item.get("sma")
            if sma:
                item["sma"] = _clean_points(sma)
            clean_chunk.append(item)
        if not clean_chunk:
            continue
        n = len(clean_chunk)
        fig, axes = plt.subplots(
            n, 1, figsize=(7.0, max(3.0, 2.1 * n)), constrained_layout=True)
        axes = [axes] if n == 1 else list(axes)

        for ax, item in zip(axes, clean_chunk):
            pts = item.get("points") or []
            xs: List[datetime] = []
            if len(pts) >= 2:
                xs = [datetime.fromtimestamp(float(ts)) for ts, _ in pts]
                ax.plot(xs, [float(p) for _, p in pts],
                        lw=1.5, color="#1f77b4", label="цена")
            sma = item.get("sma") or []
            if len(sma) >= 2:
                ax.plot([datetime.fromtimestamp(float(ts)) for ts, _ in sma],
                        [float(v) for _, v in sma],
                        lw=1.1, ls="--", color="#ff7f0e", label="средняя")
            pos = (positions or {}).get(str(item.get("ticker", "")))
            entry = float((pos or {}).get("entry_price", 0) or 0)
            if entry > 0 and pts:
                ax.axhline(entry, ls=":", lw=1.2, color="#d62728")
            title = str(item.get("ticker", "")).strip()
            subtitle = str(item.get("subtitle") or "").strip()
            ax.set_title(f"{title}  {subtitle}".strip(), fontsize=10, loc="left")
            ax.set_ylabel("цена, ₽", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            if xs:
                for label in ax.get_xticklabels():
                    label.set_rotation(20)
                    label.set_ha("right")

        if suptitle:
            fig.suptitle(suptitle, fontsize=11)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=100)
            photos.append(buf.getvalue())
        except Exception as exc:  # noqa: BLE001 - одна картинка не решает
            log.warning("Не удалось нарисовать группу графиков: %s", exc)
        finally:
            plt.close(fig)
    return photos


def build_chart_items(history_by_ticker: Dict[str, Sequence[Tuple[float, float]]],
                      *, sma_period: int = 0, subtitles: Optional[Dict[str, str]] = None
                      ) -> List[Dict[str, Any]]:
    """
    Упаковка данных в формат render_group_pngs: {ticker -> [(ts, price)]} ->
    [{"ticker", "points", "sma", "subtitle"}]. sma_period > 0 добавит линию
    средней; subtitles - приписки к заголовкам (например, изменение цены).
    """
    items: List[Dict[str, Any]] = []
    for ticker, points in history_by_ticker.items():
        pts = _clean_points(points)
        if len(pts) < 2:
            continue
        item: Dict[str, Any] = {"ticker": ticker, "points": pts, "sma": None}
        if sma_period and len(pts) >= sma_period:
            item["sma"] = sma_series(pts, sma_period)
        sub = (subtitles or {}).get(ticker)
        if sub:
            item["subtitle"] = sub
        items.append(item)
    return items


def render_forecast_png(history: Sequence[Tuple[float, float]],
                        fc: Dict[str, Any], ticker: str,
                        subtitle: str = "") -> Optional[bytes]:
    """
    График прогноза (v1.7.2, команда /predict): история цены + медианный
    путь прогноза (оранжевый пунктир) + конус 80% (заливка) на горизонте.

    history - [(ts, price)] прошлые свечи; fc - словарь из predictor.forecast
    (ключи path/band_low/band_high). matplotlib недоступен -> None.
    """
    pts = _clean_points(history)
    path = fc.get("path") or []
    low = fc.get("band_low") or []
    high = fc.get("band_high") or []
    if len(pts) < 2 or len(path) < 3:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib недоступен, график прогноза не строю: %s", exc)
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    xs = [datetime.fromtimestamp(ts) for ts, _ in pts]
    ax.plot(xs, [p for _, p in pts], lw=1.5, color="#1f77b4", label="цена (факт)")
    fx = [datetime.fromtimestamp(ts) for ts, _ in path]
    if len(low) == len(path) and len(high) == len(path):
        ax.fill_between(fx, [p for _, p in low], [p for _, p in high],
                        color="#2ca02c", alpha=0.18,
                        label="конус 80% (куда попадёт цена)")
    ax.plot(fx, [p for _, p in path], lw=1.7, ls="--", color="#2ca02c",
            label="прогноз (среднее ожидание)")
    ax.axvline(fx[0], lw=0.9, color="#888888", alpha=0.6)
    title = f"{ticker} - прогноз на {fc.get('horizon_h', 0):g} ч"
    if subtitle:
        title += f"  {subtitle}"
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("цена, ₽", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    ax.legend(fontsize=8, loc="best")
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=100)
    finally:
        plt.close(fig)
    return buf.getvalue()
