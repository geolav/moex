# TEST REPORT v1.17.2

Окружение: Python 3.12, офлайн, официальные юнит-тесты + dry_run_test.py.

| Набор | Результат |
|---|---|
| test_scalp_price_only.py (новый) | OK, 10 тестов |
| test_scalp_book_fallback.py (новый) | OK, 6 тестов |
| test_scalp_microstructure.py | OK, 17 тестов |
| test_scalp_engine_stream.py | OK, 13 тестов |
| test_scalp_stream.py | OK |
| test_scanner_scalp_ai.py | OK |
| test_trade_costs_consistency.py | OK |
| test_charts_dense_history.py | OK |
| dry_run_test.py | 152/152 |

Ключевые новые проверки:
1. **Сценарий владельца**: 120 тиков с локальной ямой, `book_age=inf`,
   спреда нет → `detect_multiscale_dip_trigger` возвращает сигнал
   `dip_reversal` c `book_mode=price_only` (раньше: None, «стакан устарел»).
2. `SCALP_MICRO_PRICE_ONLY=false` → прежнее fail-closed поведение сохранено.
3. REST-стакан (book_age 20 c) → полный режим с проверкой дисбаланса;
   устаревший REST-стакан (120 c) → корректная деградация в price-only.
4. Устаревшая цена → отказ («последняя цена устарела») — свежесть цены
   осталась жёсткой.
5. Отказ подписки на стакан (LIMIT_IS_EXCEEDED) логируется WARNING с тикером;
   SUCCESS логируется один раз на подключение.
6. REST-фолбэк заполняет bid/ask/спред, не затирает живой стримовый стакан,
   тротлится и молча переживает ошибку сети.
7. Движок переживает конфиг с None в новых полях (dry-run), 152/152.
