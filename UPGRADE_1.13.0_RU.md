# Обновление до MOEX Trading Bot v1.13.0

## 1. Перед обновлением

1. Выключи автоторговлю командой `/toggle_trading` и останови процесс.

```bash
cd ~/trading_bot/moex
pkill -f "python.*main.py"; sleep 3
mkdir -p ~/moex_backup_1.13.0
cp -a .env state.json data logs ~/moex_backup_1.13.0/ 2>/dev/null || true
```

## 2. Установка

Загрузи `moex_trading_bot_v1.13.0.zip` на сервер в `~/trading_bot/`, затем:

```bash
cd ~/trading_bot
unzip -o moex_trading_bot_v1.13.0.zip
cp -a ~/trading_bot/moex_trading_bot/. ~/trading_bot/moex/
cd ~/trading_bot/moex
```

Пользовательские `.env`, `state.json`, `data/` и `logs/` не заменяются —
архив их не содержит.

## 3. Зависимости не изменились

`requirements.txt` тот же, что в v1.12.0 (t-tech-investments с extra-index).
Проверка на всякий случай:

```bash
venv/bin/pip install -r requirements.txt
```

## 4. Настройки `.env`

Всё новое — опционально, без правок `.env` бот работает с новыми дефолтами.
Если хочешь настроить (не добавляй дубли ключей):

```dotenv
# Режим рынка: ниже - входы запрещены, порог..55% - вход с риском x0.5
MARKET_BREADTH_RISKOFF_PCT=30
# Пороги локального микро-детектора потока
SCALP_MICRO_MIN_TICKS=6
SCALP_MICRO_MIN_UP_RATIO=0.55
SCALP_MICRO_MIN_IMBALANCE=1.10
SCALP_MICRO_MAX_RET45_PCT=0.45
# Быстрее оборот (старые значения 45/20/120)
SCALP_TIME_STOP_MIN=30
SCALP_COOLDOWN_MIN=10
SCALP_STOP_COOLDOWN_MIN=45
```

Если в старом `.env` уже заданы `SCALP_TIME_STOP_MIN=45`,
`SCALP_COOLDOWN_MIN=20`, `SCALP_STOP_COOLDOWN_MIN=120` — они ПЕРЕБЬЮТ новые
дефолты. Обнови эти три строки на значения выше, если хочешь более быстрый
оборот сделки.

## 5. Проверка и запуск

```bash
venv/bin/python -m py_compile *.py
venv/bin/python dry_run_test.py          # ожидание: 120 проверок, 0 провалов
venv/bin/python -m unittest test_scalp_microstructure test_scalp_stream test_scalp_engine_stream -v
pkill -f "python.*main.py"; sleep 3
nohup venv/bin/python main.py >> logs/stdout.log 2>&1 &
tail -f logs/bot.log
```

## 6. Что смотреть в логе после запуска

- `Сканер: shortlist пуст; ... сетап: 0` — в скальпе это НЕ проблема: сканер
  только обновляет watchlist, входы идут через поток.
- `SBER: свечной сетап молчит - вход по свежему микро-импульсу потока` —
  новый путь входа заработал.
- `SBER: breadth 32.0% - нейтральная зона, вход с половинным риском` —
  режим рынка разрешает сделки с риском x0.5.
- `локальные фильтры пройдены; кандидат добавлен в очередь Groq batch` —
  кандидат уехал в ИИ; далее `Пакетный ИИ-анализ: обработано N кандидатов`.
- Спам `позиция нормализована` каждые 30 секунд исчез; сообщение появляется
  только при реальном изменении позиции.
- Ошибка 429 от Groq означает превышение тарифа — ботовский бюджет 120К
  токенов обычно срабатывает раньше, но лимиты тарифа смотри в
  Groq Console -> Limits (типовые: 200К ток/день, 8К ток/мин, 30 req/мин).
