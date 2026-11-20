# Обновление до v1.8.1

Релиз оптимизирует число обращений к Groq и поднимает лимит скальпа до 8 позиций.
Перед заменой файлов остановите сервис и сохраните свои данные.

## 1. Резервная копия и обновление файлов

```bash
sudo systemctl stop trading_bot
cd /путь/к/moex-trading-bot
cp -a .env .env.backup.$(date +%Y%m%d-%H%M%S)
[ ! -f state.json ] || cp -a state.json state.json.backup.$(date +%Y%m%d-%H%M%S)
[ ! -d data ] || cp -a data data.backup.$(date +%Y%m%d-%H%M%S)
[ ! -d logs ] || cp -a logs logs.backup.$(date +%Y%m%d-%H%M%S)
unzip -o moex_trading_bot_v1.8.1.zip
```

Не удаляйте старые `state.json`, `data/`, `logs/` и не заменяйте свой `.env` файлом
`.env.example`: в них находятся настройки, позиции, журнал и счётчик токенов.

## 2. Обновить параметры в существующем .env

Старый `.env` перекрывает значения из `.env.example` и значения по умолчанию.
Выполните из каталога проекта:

```bash
set_env() {
  key="$1"; value="$2"
  if grep -qE "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    echo "${key}=${value}" >> .env
  fi
}

set_env USE_SANDBOX True
set_env SCALP_MAX_POSITIONS 8
set_env SCALP_MAX_ENTRIES_PER_DAY 40
set_env SCALP_RISK_PER_TRADE_PCT 0.05
set_env SCALP_MAX_PORTFOLIO_RISK_PCT 0.40
set_env SCALP_MAX_POSITION_PCT 15
set_env SCALP_DAILY_LOSS_LIMIT_PCT 1.0
set_env SCAN_INTERVAL_MIN 10
set_env SCAN_DEEP_N 20
set_env SCAN_LLM True
set_env SCAN_AI_CANDIDATES 3
set_env SCAN_MIN_CANDIDATE_RANK 48
set_env SCAN_AUTO_BUY_PER_SCAN 2
set_env AI_BATCH_REQUIRED True
set_env AI_MIN_REQUEST_INTERVAL_SEC 600
set_env GROQ_DAILY_TOKEN_BUDGET 120000
set_env AUDITOR_LLM_ENABLED False
```

`GROQ_DAILY_TOKEN_BUDGET=120000` — локальный предел именно этого бота. Он оставляет
около 80 000 токенов от наблюдавшейся квоты 200 000 TPD на другие приложения,
использующие ту же организацию Groq. Это резерв, а не гарантия: если другие
приложения вместе израсходуют более 80 000 токенов, общий лимит всё равно может
быть достигнут раньше. Локальный счётчик хранится в `data/groq_usage.json`.
Не удаляйте этот файл между перезапусками.

При `SCAN_INTERVAL_MIN=10` пакетная оценка обычно происходит не чаще одного раза
за скан. В запрос попадают максимум 3 кандидата из пула до 20 свечных анализов;
если ни одна бумага не прошла технические фильтры, Groq вообще не вызывается.
Ручные `/ask` также расходуют общий budget, а повторные `/scan` в пределах
600 секунд не создают новый пакетный вызов.

## 3. Проверить окружение и запустить

```bash
# Активируйте то же виртуальное окружение, которое использует systemd.
source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile *.py
python dry_run_test.py
sudo systemctl restart trading_bot
sudo journalctl -u trading_bot -n 100 --no-pager
```

После запуска проверьте в Telegram `/status` и `/positions`. На первом запуске
оставьте `/autobuy off`, сверите позиции бота с портфелем песочницы и только после
этого включайте автопокупку.

## 4. Что означает лимит скальпа

8 — максимум одновременных позиций, а не обязательное количество сделок. Реальное
число будет ниже, если не хватает свободного капитала/рискового бюджета, если
сработали sector cap, spread, cooldown, временные фильтры или дневной stop-loss.
При риске 0,05% на вход и общем лимите 0,40% восемь одинаково рискованных позиций
могут занять весь разрешённый скальп-риск. Лимит 40 — максимум новых входов за день;
он не отменяет дневной стоп и другие предохранители.

Пакет Groq не является источником стопов и не может отменить локальные ограничения.
При недоступности модели или исчерпании token budget автоматическая покупка
остаётся запрещённой (fail-closed).
