# Отчёт о проверке v1.17.1

- `python -m py_compile *.py` — PASS.
- `test_scalp_microstructure.py` — PASS.
- `test_scalp_engine_stream.py` — PASS.
- `test_scanner_scalp_ai.py` — PASS.
- `test_charts_dense_history.py` — PASS.
- `test_trade_costs_consistency.py` — PASS.
- `test_scalp_stream.py` — PASS.
- Контроль готовности и свежести stream проверен тестами `test_scalp_engine_stream.py`; при недоступном потоке тест подтверждает безопасный REST fallback и явное WARNING-сообщение.
- Всего целевых unit/regression тестов: 36/36 PASS.
- Полный `dry_run_test.py` проверен в offline-режиме с заглушкой только для импортов Groq (сетевые вызовы не выполнялись): 140 PASS, 12 FAIL. Провалы относятся к ранним legacy-сценариям обычного режима (группы первых тестов); целевые scalp/regression тесты выше проходят. В обычном окружении тест также требует установленные зависимости `groq` и Telegram-пакет `aiogram`.
- Реальные брокерские и Groq API вызовы не выполнялись.
