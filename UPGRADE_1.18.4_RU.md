# Обновление MOEX Trading Bot до v1.18.4

## 1. Остановить бота и создать backup

В Telegram отправьте `/shutdown`. Затем на VPS:

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
```

## 2. Передать архив

На своём компьютере:

```bash
scp MOEX_bot_v1_18_4_local_learning.zip root@168.113.189.205:/root/
```

## 3. Обновить исходники, сохранив свои данные

На VPS:

```bash
cd /root/trading_bot
unzip -q /root/MOEX_bot_v1_18_4_local_learning.zip -d /root/trading_bot
cp -a /root/trading_bot/moex_1184_local_learning/. /root/trading_bot/moex/
cd /root/trading_bot/moex
```

Архив не содержит ваши `.env`, `state.json`, `data/`, `logs/` и `venv/`. Не заменяйте их файлами из архива. `supervisor.sqlite3` будет автоматически мигрирована при запуске; старые данные не удаляются.

## 4. Включить локальный backend

Откройте существующий файл:

```bash
nano /root/trading_bot/moex/.env
```

Добавьте или измените значения (не создавайте дублирующие строки):

```dotenv
DECISION_BACKEND=local
LOCAL_MODEL_MIN_SAMPLES=240
LOCAL_MODEL_MIN_CLASS_SAMPLES=40
LOCAL_MODEL_MIN_AUC=0.53
LOCAL_MODEL_MIN_PROBABILITY=0.62
LOCAL_MODEL_MIN_EDGE_MULTIPLE=1.25
LOCAL_WATCHLIST_MIN_RANK=58

SCALP_QUICK_TP=false
SCALP_STOP_CONFIRM_SEC=4
SCALP_STOP_CONFIRM_TICKS=3
SCALP_STOP_CONTINUE_PCT=0.03
SCALP_STOP_RECLAIM_PCT=0.05
SCALP_STOP_EMERGENCY_EXTRA_PCT=0.10
SCALP_MAX_POSITIONS=15
SCALP_WATCHLIST_MAX=25
SCALP_WATCHLIST_MAX_PER_SECTOR=4
SCALP_MAX_SECTOR_POSITIONS=2
SCALP_RISK_PER_TRADE_PCT=0.025
SCALP_MAX_PORTFOLIO_RISK_PCT=0.40
SCALP_MAX_ENTRIES_PER_DAY=100
SCALP_MIN_NET_PROFIT_RUB=1.0
SCALP_FORCE_FLAT_AT_SESSION_END=false
SCAN_DEEP_N=35
SCAN_AI_CANDIDATES=3
SUPERVISOR_ENABLED=true
SUPERVISOR_AUTO_TUNE=true
SUPERVISOR_GROQ_ENABLED=false
```

Остальные ваши настройки брокера, Telegram, прокси и секреты оставьте без изменений. Не заменяйте весь `.env` файлом `.env.example`.

## 5. Проверить и запустить

```bash
cd /root/trading_bot/moex
venv/bin/python -m compileall -q .
venv/bin/python -m unittest discover -p 'test_*.py' -v
venv/bin/python main.py
```

В старте ожидается `MOEX Trading Bot v1.18.4` и строка `Локальная ML-модель активна ... сетевые AI-запросы отключены`. Supervisor должен показывать `backend=local`. Модель при первом запуске может быть `ready=False` — это нормально. Она учится на свежих observations; не пытайтесь вручную выставить `ready=true`.

## 6. Как вернуть Groq

В `.env` измените одну строку:

```dotenv
DECISION_BACKEND=groq
```

И перезапустите бота. Модель продолжит храниться в `data/local_directional_model.json`, а Supervisor продолжит её обучать, если включён. Если вы ранее отдельно устанавливали `LLM_DECISIONS=false`, верните `LLM_DECISIONS=true`, чтобы снова разрешить Groq. Для работы Groq должен быть доступен сам API endpoint: если прежний Deno-прокси исчерпал лимиты, очистите `GROQ_BASE_URL` для прямого API (при доступности) или укажите рабочий прокси.

## Важные замечания

- Local backend предназначен для scalp event-trigger BUY/HOLD. Обычная стратегия без Groq закрывается fail-closed.
- Пока модель прогревается, в песочнице допускается прежний строгий fast-track с 25% размера; на реальном счёте новых входов до готовности модели нет.
- После готовности каждая покупка проходит локальную модель и прежние математические, рыночные и risk-фильтры.
- Модель не управляет продажами и не отменяет аварийный стоп. Сначала проверяйте на песочнице.
- Перенос позиций через ночь по-прежнему возможен при `SCALP_FORCE_FLAT_AT_SESSION_END=false`.
