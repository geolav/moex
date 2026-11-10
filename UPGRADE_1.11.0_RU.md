# Обновление до MOEX Trading Bot v1.11.0

## Важно

Этот архив собран на основе доступного v1.9.0 с переносом изменений v1.10.0 по описанию владельца. Это не точное объединение с исходным кодом v1.10.0, который не был предоставлен.

## 1. Перед обновлением

1. В Telegram выключи автоторговлю (`/toggle_trading`), проверь, что она выключена, и останови процесс через `/shutdown` или `Ctrl+C`.
2. Сделай резервную копию `.env`, `state.json`, `data/` и `logs/`:

```bash
cd ~/trading_bot/moex
mkdir -p ~/moex_backup_1.11.0
cp -a .env state.json data logs ~/moex_backup_1.11.0/ 2>/dev/null || true
```

## 2. Распаковка

Загрузи `moex_trading_bot_v1.11.0.zip` в `~/trading_bot/`. В архиве файлы находятся внутри каталога `moex_trading_bot/`, поэтому обнови рабочую папку так:

```bash
cd ~/trading_bot
unzip -o moex_trading_bot_v1.11.0.zip
cp -a ~/trading_bot/moex_trading_bot/. ~/trading_bot/moex/
```

Если твой рабочий каталог называется иначе, замени `~/trading_bot/moex/` на фактический путь.

Не заменяй пользовательские `.env`, `state.json`, `data/`, `logs/`. В ZIP их нет.

## 3. Проверка кода

```bash
cd ~/trading_bot/moex
venv/bin/python -m py_compile *.py
venv/bin/python dry_run_test.py
```

Ожидаемо: `ИТОГО: пройдено 109, провалено 0`.

## 4. Настройки

В существующем `.env` проверь/добавь (не добавляй дубли, если параметры уже есть):

```dotenv
SCALP_MAX_POSITIONS=8
SCALP_WATCHLIST_MAX=10
SCALP_TIME_STOP_MIN=45
SCALP_RISK_PER_TRADE_PCT=0.05
SCALP_MAX_PORTFOLIO_RISK_PCT=0.40
SCALP_DAILY_LOSS_LIMIT_PCT=1.0
SCALP_MAX_ENTRIES_PER_DAY=40
```

Остальные старые пользовательские значения `.env` сохраняются. Значения `.env` перекрывают defaults в коде.

## 5. Первый запуск

```bash
cd ~/trading_bot/moex
venv/bin/python main.py
```

Сначала проверь `/status`, `/positions`, `/memory` и логи запуска. Убедись, что старые позиции мигрированы, не осталось расхождений state/брокер и нет повторяющихся исключений в `engine`. Только после проверки включай `/toggle_trading`.

Рекомендуется первые сессии оставить песочницу и выключить автопокупку через `/autobuy off`, чтобы проверить входы вручную. Переходить на реальные деньги по итогам dry-run тестов нельзя: тесты проверяют программные сценарии, а не прибыльность стратегии.
