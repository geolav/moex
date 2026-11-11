# Обновление до MOEX Bot 1.18.1

## 1. Перед обновлением

На VPS останови процесс через Telegram `/shutdown` (или останови systemd-службу, если используешь её). Затем:

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
```

Резервная копия сохраняет исходный проект целиком.

## 2. Перенеси архив

На своём компьютере:

```bash
scp MOEX_bot_v1_18_1_active_supervisor.zip root@168.113.189.205:/root/
```

На сервере:

```bash
cd /root/trading_bot
unzip -q /root/MOEX_bot_v1_18_1_active_supervisor.zip -d /root/trading_bot/
cp -a /root/trading_bot/moex_1181_active/. /root/trading_bot/moex/
```

Копируются файлы программы. Существующие `.env`, `state.json`, `data/`, `logs/` и `venv/` не удаляются.

## 3. Добавь настройки

Открой:

```bash
nano /root/trading_bot/moex/.env
```

Добавь, если таких переменных ещё нет:

```dotenv
# Длинная история изменений цены для локальных ям 12-240 секунд
SCALP_TICK_BUFFER_MAXLEN=2400

# Наблюдение и обучение
SUPERVISOR_ENABLED=true
SUPERVISOR_INTERVAL_SEC=60
SUPERVISOR_LOOKAHEAD_SEC=300
SUPERVISOR_GROQ_ENABLED=true
SUPERVISOR_GROQ_MIN_INTERVAL_SEC=3600
SUPERVISOR_AUTO_TUNE=true
SUPERVISOR_GENERATE_PATCHES=true

# Резервные ценовые снимки; одинаковые цены не считаются новым тиком
SCALP_PRICE_REST_REFRESH_SEC=2.5
SCALP_PRICE_STALE_SEC=2.5
```

Существующий `GROQ_API_KEY`, ключ T-Invest, Telegram-токен и остальные настройки не заменяй. Не присылай `.env` в чат.

## 4. Проверка и запуск

```bash
cd /root/trading_bot/moex
venv/bin/python -m py_compile *.py
venv/bin/python -m pytest -q --ignore=dry_run_test.py
venv/bin/python main.py
```

Полный `dry_run_test.py` отдельно требует установленный пакет `groq` (он указан в `requirements.txt`). Не запускай `pip install` вслепую на VPS, если пакет уже присутствует в рабочем `venv`.

## 5. Обязательно проверь автоторговлю

В логах посмотри строку:

```text
Режим: ПЕСОЧНИЦА | торговля: вкл | ...
```

Если там `торговля: выкл`, автоматических покупок НЕ будет. В Telegram отправь:

```text
/trading_on
```

Это отдельная команда включения, она не переключает ВКЛ обратно в ВЫКЛ. Убедись, что режим — ПЕСОЧНИЦА. Не включай автоторговлю в реальном режиме, пока не проверил работу в песочнице.

`/autobuy` — отдельный переключатель автопокупки кандидатов обычного сканера. Он не заменяет `trading_enabled` и не включает сам по себе исполнение локальных скальп-входов.

## 6. Что смотреть в логах

Раз в минуту ожидай примерно такую строку:

```text
Supervisor: снимки stream=... REST=...; реальные изменения цены за 8с=...; оценок детектора/ч=...; триггеров=...; теневых ям=...; ... автоторговля=ВКЛ/ВЫКЛ
```

Примеры новых сообщений:

```text
Микродно SBER: оценка завершена; ... основной детектор: ...; теневой: ...
Supervisor shadow SBER: локальная яма-кандидат; ... проверю результат через 300 сек
Supervisor: SBER shadow найден, но вход заблокирован: trading_disabled
```

В Telegram:

```text
/supervisor
```

В отчёте различаются свежий REST-снимок и реальное изменение рыночной цены. После закрытия биржи REST может отвечать успешно, но реальных новых ценовых изменений не будет.
