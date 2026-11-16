# Обновление MOEX Trading Bot до v1.18.6

## 1. Сделать резервную копию

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
```

Остановите службу бота перед заменой файлов.

## 2. Передать и распаковать архив

На локальном компьютере:

```bash
scp MOEX_bot_v1_18_6_local_learning.zip root@YOUR_SERVER:/root/
```

На VPS:

```bash
cd /root/trading_bot
unzip -q /root/MOEX_bot_v1_18_6_local_learning.zip
cp -a /root/trading_bot/moex_1186_local_learning/. /root/trading_bot/moex/
cd /root/trading_bot/moex
```

Архив содержит только исходники, тесты и документацию. Он не содержит `.env`, `state.json`, `data/`, `logs/`, `venv/` или брокерских/Telegram-токенов. Не удаляйте эти рабочие данные.

## 3. Настройки .env

В твоём текущем `.env` уже выставлено `DECISION_BACKEND=local`. Эту строку оставь.

Чтобы Supervisor продолжал собирать данные, обучать модель и осторожно адаптировать пороги, сохрани:

```dotenv
SUPERVISOR_ENABLED=true
SUPERVISOR_GROQ_ENABLED=false
SUPERVISOR_AUTO_TUNE=true
SUPERVISOR_GENERATE_PATCHES=true
SUPERVISOR_INTERVAL_SEC=60
SUPERVISOR_LOOKAHEAD_SEC=300
SUPERVISOR_TARGET_COST_MULTIPLE=1.25
```

Добавь только если соответствующих переменных ещё нет:

```dotenv
LOCAL_MODEL_MIN_SAMPLES=200
LOCAL_MODEL_MIN_CLASS_SAMPLES=40
LOCAL_MODEL_MIN_AUC=0.53
LOCAL_MODEL_MIN_PROBABILITY=0.62
LOCAL_MODEL_MIN_EDGE_MULTIPLE=1.25
SUPERVISOR_TUNE_MIN_EPISODES=40
```

Не создавай дублирующиеся строки. `SUPERVISOR_AUTO_TUNE=true` теперь безопаснее прежнего: минимум 40 разреженных эпизодов, минимум 8 тикеров, минимум час наблюдений, статистический интервал net expectancy и шестичасовая пауза между реальными изменениями. `SUPERVISOR_GENERATE_PATCHES=true` оставляет предложения в staging; живые файлы автоматически не перезаписываются.

## 4. Данные обучения и старая политика

**Не удаляй**:

- `data/supervisor.sqlite3` — историю наблюдений, теневых сценариев и сделок;
- `data/local_directional_model.json` — веса модели;
- `data/trade_memory.json` и `state.json` — память и состояние бота.

Файл старой политики `data/supervisor/learned_policy.json` v1 вручную удалять не требуется. Код v1.18.6 автоматически игнорирует её старые пороги и возвращается к базовым значениям стратегии. После накопления новых независимых данных появится политика v2.

## 5. Проверка и запуск

```bash
cd /root/trading_bot/moex
venv/bin/python -m compileall -q .
venv/bin/python -m pytest -q --ignore=dry_run_test.py
```

Запусти сервис обычным способом и проверь:

- в старте: `Backend решений: Local ML` и `сетевые AI-запросы отключены`;
- Supervisor: `backend=local`, `auto_tune=True`;
- периодические логи `Supervisor ML` показывают число target/HOLD и причину `ready=False`;
- при блокировке новостей в логе теперь присутствуют конкретные RSS-заголовки.

В Telegram используй `/supervisor`. Рост количества примеров сам по себе не означает готовность: `ready=True` появляется только после проверки AUC и Brier. Если AUC остаётся 0.500 или Brier не лучше базового прогноза, модель продолжит HOLD.

Оставь `USE_SANDBOX=True`. Нельзя гарантировать прибыль, и этот релиз не доказывает прибыльность стратегии. Сначала оценивай реальные сделки и net P&L после полных издержек на песочнице.
