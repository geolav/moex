# Обновление до MOEX Trading Bot v1.12.0

## 1. Перед обновлением

1. Выключи автоторговлю командой `/toggle_trading` и останови процесс (`Ctrl+C` или `/shutdown`).
2. Сохрани пользовательские настройки и состояние:

```bash
cd ~/trading_bot/moex
mkdir -p ~/moex_backup_1.12.0
cp -a .env state.json data logs ~/moex_backup_1.12.0/ 2>/dev/null || true
```

## 2. Установка

Распакуй архив, не заменяя пользовательские `.env`, `state.json`, `data/` и `logs/`:

```bash
cd ~/trading_bot
unzip -o moex_trading_bot_v1.12.0.zip
cp -a ~/trading_bot/moex_trading_bot/. ~/trading_bot/moex/
cd ~/trading_bot/moex
```

## 3. Обновление зависимостей

В активном venv:

```bash
venv/bin/pip install -r requirements.txt
```

Для stream-скальпа нужен официальный `t-tech-investments` из extra-index в `requirements.txt`; без SDK или без доступного market-data stream новые scalp-входы будут заблокированы.

## 4. Настройки `.env`

Проверь, что присутствуют (не добавляй дубли). Если в старом `.env` указана `llama-3.1-8b-instant`, замени её на `openai/gpt-oss-20b`: старая модель выведена из эксплуатации.

```dotenv
SCALP_MAX_POSITIONS=8
SCALP_WATCHLIST_MAX=10
SCALP_TIME_STOP_MIN=45
SCALP_RISK_PER_TRADE_PCT=0.05
SCALP_MAX_PORTFOLIO_RISK_PCT=0.40
SCALP_DAILY_LOSS_LIMIT_PCT=1.0
SCALP_MAX_ENTRIES_PER_DAY=40
SCALP_EXIT_POLL_SEC=30
SCALP_AI_MIN_REQUEST_INTERVAL_SEC=240
GROQ_MODEL=openai/gpt-oss-20b
GROQ_DAILY_TOKEN_BUDGET=120000
```

`SCALP_POLL_SEC` оставлен для совместимости, но больше не определяет частоту поиска входов. Поток цены/стакана event-driven. Groq получает до трёх лучших локальных кандидатов не чаще одного пакета за интервал `SCALP_AI_MIN_REQUEST_INTERVAL_SEC` (по умолчанию 240 секунд).

## 5. Проверка и запуск

```bash
venv/bin/python -m py_compile *.py
venv/bin/python -m unittest test_scalp_microstructure test_scalp_stream test_scalp_engine_stream -v
venv/bin/python dry_run_test.py
venv/bin/python main.py
```

Сначала оставь песочницу включённой и автоторговлю выключенной. Проверь логи `Scalp stream: подписываюсь...`, потоковые события, отсутствие ошибок SDK и статистику Groq. При недоступном stream бот намеренно не открывает новые scalp-позиции; это безопаснее, чем незаметно перейти на медленный REST-вход.

Первые реальные сделки не запускай только на основании dry-run: он проверяет программную логику на моках, но не подтверждает качество исполнения или прибыльность стратегии.
