# Hotfix v1.8.2 — исправление запуска сканера

Ошибка из лога:

```text
TypeError: MarketScanner._deep_analyze() got an unexpected keyword argument 'breadth_pct'
```

Исправлена сигнатура метода в `scanner.py`. Остальная торговая логика и формат
`state.json`, `data/`, `logs/`, `.env` не меняются.

## Обновление на сервере

Из каталога проекта (`~/trading_bot/moex` в приведённом логе):

```bash
cp scanner.py scanner.py.bak.$(date +%Y%m%d-%H%M%S)
unzip -o moex_trading_bot_v1.8.2.zip scanner.py telegram_bot.py dry_run_test.py README.md CHANGELOG_1.8.2_RU.md HOTFIX_1.8.2_RU.md
venv/bin/python -m py_compile scanner.py telegram_bot.py
venv/bin/python dry_run_test.py
```

Если архив распакован в другой каталог, укажите его путь после `unzip -o`.

Затем перезапустите процесс бота. При ручном запуске остановите старый процесс
`Ctrl+C` и выполните `venv/bin/python main.py`. Для systemd используйте
`sudo systemctl restart trading_bot`.

После запуска в логах должны появиться строки о количестве глубоко
проанализированных бумаг и о числе кандидатов, отправленных ИИ. Ошибки
`unexpected keyword argument 'breadth_pct'` больше быть не должно.
