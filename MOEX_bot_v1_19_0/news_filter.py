"""Локальное сопоставление новостей с эмитентом и фильтр корпоративных рисков.

Новость сначала должна быть однозначно связана с конкретным эмитентом
(тикер/официальное название/специфичный псевдоним). Только после этого
заголовок проверяется на негативное корпоративное событие. Общие слова вроде
«банк», «система» или «самолёт» не используются как самостоятельные ключи.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

# Известные однозначные названия и биржевые/медийные псевдонимы.
# Намеренно избегаем неоднозначных одиночных слов («банк», «система», «самолёт»).
_ISSUER_ALIASES = {
    "VTBR": ("втб", "vtb", "банк втб"),
    "SBER": ("сбербанк", "сбер", "sberbank", "sber"),
    "SBERP": ("сбербанк", "сбер", "sberbank", "sber"),
    "GAZP": ("газпром", "gazprom"),
    "LKOH": ("лукойл", "lukoil"),
    "YDEX": ("яндекс", "yandex"),
    "MTSS": ("мтс", "мобильные телесистемы", "mts"),
    "GMKN": ("норникель", "норильский никель", "nornickel"),
    "NVTK": ("новатэк", "novatek"),
    "X5": ("x5 group", "x5 retail group", "икс 5", "икс5"),
    "CHMF": ("северсталь", "severstal"),
    "ROSN": ("роснефть", "rosneft"),
    "OZON": ("озон", "ozon"),
    "POSI": ("позитив технолоджиз", "positive technologies", "группа позитив"),
    "SVCB": ("совкомбанк", "sovcombank"),
    "ALRS": ("алроса", "alrosa"),
    "AFLT": ("аэрофлот", "aeroflot"),
    "SMLT": ("гк самолет", "группа самолет", "девелопер самолет", "samolet"),
    "AFKS": ("афк система", "afk sistema"),
    "MGNT": ("сеть магнит", "ритейлер магнит", "группа магнит", "magnit"),
    "MOEX": ("мосбиржа", "московская биржа", "moex"),
    "RAGR": ("русагро", "rusagro"),
    "PHOR": ("фосагро", "phosagro"),
    "PLZL": ("полюс золото", "polyus"),
    "VKCO": ("вконтакте", "vk company", "vkco"),
    "DOMRF": ("дом.рф", "дом рф", "домрф"),
}

_BLOCK_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in [
        ("санкции/ограничения", r"санкц|ограничени[ея].{0,30}(экспорт|операц|деятельност|расчет)"),
        ("суд/иск/расследование", r"судебн.{0,20}(иск|разбирательств)|подал.{0,15}иск|иск.{0,20}(к компании|против)|расследован|обыск|уголовн"),
        ("дефолт/банкротство", r"дефолт|банкротств|неисполнени.{0,20}(обязательств|долг)|реструктуризац.{0,20}долг"),
        ("разводнение капитала", r"дополнительн.{0,20}эмисс|допэмисс|размещени.{0,20}(новых )?акци|конвертируем.{0,20}облигац"),
        ("листинг/торги", r"делистинг|приостановк.{0,20}торг|прекращени.{0,20}торг"),
        ("слабая отчётность", r"чист.{0,20}убыт|убыток.{0,30}(вместо|против)|снизил.{0,20}(прогноз|guidance)|ухудшил.{0,20}(прогноз|ожидани)"),
        ("дивидендный негатив", r"отказ.{0,20}от.{0,10}дивиденд|отмен.{0,20}дивиденд|не рекоменд.{0,30}дивиденд"),
        ("fraud/default/legal", r"bankrupt|default|delisting|sanction|fraud|investigation|lawsuit|downgrade|going concern"),
    ]
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold().replace("ё", "е")).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalise(phrase)
    if not phrase:
        return False
    # Границы слова не дают VTBR совпасть с частью другого слова.
    pattern = re.compile(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", re.UNICODE)
    return bool(pattern.search(text))


def news_matches_issuer(text: str, ticker: str, company_name: str = "") -> bool:
    """True только при явном упоминании тикера или конкретного эмитента."""
    content = _normalise(text)
    code = _normalise(ticker).upper()
    if not content or not code:
        return False
    if _contains_phrase(content, code):
        return True

    terms = list(_ISSUER_ALIASES.get(code, ()))
    full_name = _normalise(company_name)
    # Полное название эмитента допускается целиком. Первое слово названия
    # отдельно никогда не используется: это и давало ложный поиск «Банк».
    if len(full_name) >= 5 and full_name not in {"банк", "группа", "компания", "система"}:
        terms.append(full_name)
    return any(_contains_phrase(content, term) for term in dict.fromkeys(terms))


def matching_risk_headlines(headlines: Iterable[str]) -> List[Tuple[str, str]]:
    """Возвращает (причина, конкретный заголовок/элемент RSS) для аудита."""
    matches: List[Tuple[str, str]] = []
    for item in headlines:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        for label, pattern in _BLOCK_PATTERNS:
            if pattern.search(text):
                matches.append((label, text[:240]))
    return matches


def classify_headlines(headlines: Iterable[str]) -> Tuple[float, List[str]]:
    """-0.70 при явном негативном событии в одном релевантном элементе RSS.

    Каждый элемент анализируется отдельно: слова из двух независимых статей
    не могут случайно сложиться в один risk-pattern.
    """
    matches = matching_risk_headlines(headlines)
    reasons = list(dict.fromkeys(label for label, _ in matches))
    return (-0.70 if reasons else 0.0), reasons
