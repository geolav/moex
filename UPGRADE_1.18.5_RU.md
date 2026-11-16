# Обновление MOEX Trading Bot до v1.18.5

## 1. Остановить бота и сделать резервную копию

```bash
cd /root/trading_bot
cp -a moex "moex_backup_$(date +%Y%m%d_%H%M%S)"
```

## 2. Загрузить и распаковать архив

На компьютере:

```bash
scp MOEX_bot_v1_18_5_local_first.zip root@YOUR_SERVER:/root/
```

На VPS:

```bash
cd /root/trading_bot
unzip -q /root/MOEX_bot_v1_18_5_local_first.zip
cp -a /root/trading_bot/moex_1185_local_first/. /root/trading_bot/moex/
cd /root/trading_bot/moex
```

Архив не содержит `.env`, `state.json`, `data/`, `logs/` и `venv/`. Не удаляйте и не заменяйте эти каталоги.

## 3. Обновить только необходимые строки в существующем .env

```bash
nano /root/trading_bot/moex/.env
```

Добавьте одну строку (если её нет):

```dotenv
DECISION_BACKEND=local
```

Измените существующие строки:

```dotenv
SUPERVISOR_GROQ_ENABLED=false
SUPERVISOR_AUTO_TUNE=false
SUPERVISOR_GENERATE_PATCHES=false
```

Остальные настройки, включая `TRADING_ENABLED`, sandbox и параметры стратегии, оставьте как есть. Не создавайте дубликаты переменных. Ключ Groq можно оставить в `.env`, он не будет использоваться торговым backend при `DECISION_BACKEND=local`; для дополнительной гигиены можно удалить его из файла, но это не обязательно.

## 4. Сбросить только устаревшую автополитку Supervisor

В старом логе auto-tune довёл `dip_min_drop_pct` до верхней границы. Сначала сохраните файл, затем удалите только его, чтобы вернуться к базовым порогам из конфигурации:

```bash
cd /root/trading_bot/moex
if [ -f data/supervisor/learned_policy.json ]; then
  cp -a data/supervisor/learned_policy.json data/supervisor/learned_policy.json.bak
  rm data/supervisor/learned_policy.json
fi
```

**Не удаляйте** `data/supervisor.sqlite3` и `data/local_directional_model.json`: там история наблюдений, метки, фактические сделки и обученные веса.

## 5. Проверить запуск

```bash
cd /root/trading_bot/moex
venv/bin/python -m compileall -q .
venv/bin/python -m unittest discover -p 'test_*.py' -v
```

Запустите сервис обычным способом. В логе должны быть строки о `DECISION_BACKEND=local`, `Локальная ML-модель активна ... сетевые AI-запросы отключены` и `backend=local`. Не должно быть `AIAnalyst готов` с Groq-моделью или ожиданий Groq quota.

## 6. Что ожидать от покупок

По журналу предыдущего запуска у модели было 182 размеченных примера при требовании 240; распределение было 137 положительных и 45 отрицательных, а holdout AUC — 0.500. Поэтому после переключения на local она не обязана немедленно начать обычные покупки: сначала нужно накопить достаточную выборку и пройти временную проверку качества. До готовности модели в песочнице допускается только существующий строгий fast-track с четвертью объёма; он тоже не покупает, если ожидаемое движение не покрывает издержки и риск. Это защита от случайных сделок, а не зависание backend.

Проверить статус можно командой `/supervisor`. Важные признаки: `Backend решений: local`, растут `samples/up/down`, `ready=true` только после прохождения holdout-проверки. Не выставляйте `ready=true` вручную и не снижайте AUC/вероятностные пороги ради количества сделок.

**Нельзя гарантировать прибыль или обещать, что частые покупки будут прибыльными.** Сначала оставьте песочницу, проверяйте фактический net P&L после комиссий и качество модели на новых данных. Не переключайте на реальный счёт только потому, что модель стала `ready`.
