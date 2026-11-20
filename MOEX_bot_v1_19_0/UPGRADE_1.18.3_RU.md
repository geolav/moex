# Обновление до MOEX Bot 1.18.3

## 1. Остановить бота и сделать резервную копию

В Telegram отправьте `/shutdown`. На сервере:

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
```

## 2. Передать архив на VPS

На своём компьютере:

```bash
scp MOEX_bot_v1_18_3_smarter_exits.zip root@168.113.189.205:/root/
```

## 3. Обновить исходники

На VPS:

```bash
cd /root/trading_bot
unzip -q /root/MOEX_bot_v1_18_3_smarter_exits.zip -d /root/trading_bot
cp -a /root/trading_bot/moex_1183_smarter_exits/. /root/trading_bot/moex/
cd /root/trading_bot/moex
```

Архив не содержит вашего `.env`, `state.json`, `data/`, `logs/` и виртуального окружения. Команда копирует файлы новой версии поверх проекта и сохраняет эти локальные данные.

## 4. Обновить существующий .env

```bash
nano /root/trading_bot/moex/.env
```

Измените существующие значения (не добавляйте вторые строки с теми же ключами):

```dotenv
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
SCAN_DEEP_N=35
SCAN_AI_CANDIDATES=3
SCALP_FORCE_FLAT_AT_SESSION_END=false
```

**Особенно важно:** если в старом `.env` осталось `SCALP_QUICK_TP=true`, замените на `false`, иначе бот продолжит преждевременно закрывать позиции при первом net +1 ₽.

## 5. Проверить и запустить

```bash
cd /root/trading_bot/moex
venv/bin/python -m py_compile *.py
venv/bin/python -m unittest discover -p 'test_*.py' -v
venv/bin/python main.py
```

В стартовом логе проверьте `MOEX Trading Bot v1.18.3`, `до 15 позиций`, а в `Scalp режим` — выбранный дневной лимит. В `.env` и `state.json` проверьте, что автоторговля разрешена. Новые сделки сначала проверьте в песочнице.

## 6. Что смотреть в логах

- `стоп ... проколот; жду стабилизацию и повторное снижение` — первое касание стопа, позиция пока остаётся открытой.
- `после пробоя стопа цена стабилизировалась ... жду повторного снижения` — пауза подтверждена.
- `пробой стопа подтверждён` — цена после паузы обновила минимум; отправляется продажа.
- `ложный прокол стопа отменён` — цена вернулась выше уровня восстановления.
- `TP1 1R` — обычный плановый частичный выход.
- `watchlist ...` — список наблюдаемых бумаг; до 25 бумаг. Это не означает, что бот обязан открыть по ним позиции.

## Важно

Подтверждение стопа уменьшает число продаж на одиночном ложном проколе, но увеличивает время нахождения в просадке. Аварийная граница остаётся активной и может закрыть позицию в минус. Позиции без тайм-стопа могут переноситься через ночь и подвергаться гэпам. Сначала проверьте поведение на песочнице.
