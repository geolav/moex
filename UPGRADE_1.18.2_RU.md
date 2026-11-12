# Обновление до MOEX Bot 1.18.2

## Перед установкой

1. Останови бота через Telegram `/shutdown`.
2. На VPS создай резервную копию папки проекта.
3. Не удаляй и не заменяй действующие `.env`, `state.json`, `data/`, `logs/` и `venv/`.

## Копирование на VPS

На своём компьютере отправь ZIP:

```bash
scp MOEX_bot_v1_18_2_hold_profit.zip root@168.113.189.205:/root/
```

На VPS:

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
unzip -q /root/MOEX_bot_v1_18_2_hold_profit.zip -d /root/trading_bot
cp -a /root/trading_bot/moex_1182_hold_profit/. /root/trading_bot/moex/
cd /root/trading_bot/moex
venv/bin/python -m py_compile *.py
venv/bin/python main.py
```

## Обязательно проверь существующий `.env`

Поскольку старый `.env` имеет приоритет над значениями по умолчанию, добавь или замени эти строки:

```dotenv
SCALP_MAX_POSITIONS=12
SCALP_RISK_PER_TRADE_PCT=0.03
SCALP_MAX_PORTFOLIO_RISK_PCT=0.40
SCALP_MIN_NET_PROFIT_RUB=1.0
SCALP_FORCE_FLAT_AT_SESSION_END=false
```

Не присылай `.env` никому: там секретные токены. `SCALP_TIME_STOP_MIN` можешь оставить — в этой версии он legacy и не запускает продажу.

## После запуска

- Проверь в `/status`, что песочница и автоторговля включены так, как ты ожидаешь. Если выключена, в песочнице включи командой `/trading_on`.
- Убедись, что включён `/scalp`.
- В сообщении о включении скальпа должно быть видно: 12 позиций, риск 0.03%, суммарный риск ≤0.40%, фиксация от +1 ₽ net, отсутствие временного стопа.
- Проверь `/positions` и лог `Скальп-мониторинг`: там теперь печатается ожидаемый net P&L и минимальный net-порог фиксации.

**Важный риск:** `SCALP_FORCE_FLAT_AT_SESSION_END=false` означает перенос позиции через ночь. Программный стоп не является гарантированным биржевым стоп-ордером; при гэпе цена исполнения может быть хуже защитной отметки. Сначала проверь обновление в песочнице.
