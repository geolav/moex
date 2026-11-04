# MOEX Trading Bot (T-Invest API + Groq AI + Telegram)

Production-ready торговый бот для Московской биржи, спроектированный под VPS
с **1 ГБ ОЗУ** и **15 ГБ диска**. Работает через официальный T-Invest API,
использует **бесплатный** Groq API (тональность новостей + аудит логов) и
управляется из Telegram.

> Дисклеймер: алгоритмическая торговля - это риск. Бот - инструмент, а не
> гарантия прибыли. Первые недели держите его ТОЛЬКО в песочнице.

---

## 1. Архитектура

```
moex_trading_bot/
├── main.py              # точка входа: логирование, запуск всех задач, graceful shutdown
├── config.py            # конфиг из .env + StateStore (state.json: флаги, позиции)
├── tinkoff_client.py    # обёртка T-Invest API: песочница/реал, цены, свечи, ордера, RSS-новости
├── ai_analyst.py        # Groq: analyze_sentiment(news_text) -> -1.0..1.0 + аудит логов
├── trading_engine.py    # стратегия: цена < SMA-24 И сентимент > -0.3 -> BUY; SL -3% / TP +5%
├── ai_auditor.py        # каждый час: последние 50 строк лога -> Groq -> вердикт в Telegram
├── telegram_bot.py      # /start /status /toggle_trading /toggle_sandbox /toggle_auditor
├── cleanup.py           # чистка старых логов (диск 15 ГБ не переполнится)
├── dry_run_test.py      # самопроверка на моках: python dry_run_test.py
├── requirements.txt
├── .env.example         # шаблон окружения -> копируем в .env
├── trading_bot.service  # systemd unit (автозапуск + авторестарт + MemoryMax)
└── logs/                # bot.log (ротация 5 МБ x 3 файла)
```

**Стратегия** (часовой таймфрейм, CANDLE_INTERVAL_HOUR):
- **Покупка**: цена < SMA-24 (по закрытым часовым свечам) И тональность
  новостей от Groq > -0.3 И торговля включена И нет открытой позиции.
- **Продажа**: +5% (тейк-профит) или -3% (стоп-лосс) от цены входа.
- **Размер позиции**: не более 10% свободных средств, пересчёт в лоты.

**Критичные правила памяти** (выполнены):
- тикеры (макс. 5 из `TRADING_TICKERS`) обрабатываются **строго последовательно**
  в `for`-цикле; `asyncio.gather()` в коде **не используется вообще**;
- без pandas/numpy - только stdlib + лёгкие SDK; история свечей не хранится;
- `RotatingFileHandler`: 5 МБ x 3 файла (~20 МБ максимум на диске).

**Почему `t-tech-investments`, а не `tinkoff-investments`**: официальный SDK
Т-Инвестиций был переименован после ребрендинга Т-Банка. Старый пакет снят с
PyPI (pip возвращает 404). Тот же самый официальный SDK теперь называется
`t-tech-investments` и импортируется как `from t_tech.invest import ...`.
Он ставится с официального индекса T-Bank - строка `--extra-index-url` уже
прописана в `requirements.txt`.

---

## 2. Подготовка VPS (Ubuntu 22.04/24.04)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
sudo adduser --disabled-password --gecos "" botuser   # непривилегированный пользователь
sudo mkdir -p /opt/trading_bot && sudo chown botuser:botuser /opt/trading_bot
```

## 3. Установка кода и зависимостей

```bash
# Загрузите папку проекта на сервер (scp/rsync/git) в /opt/trading_bot,
# затем под пользователем botuser:
cd /opt/trading_bot
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

Если индекс T-Bank недоступен с вашего VPS, официальные зеркала SDK:
```bash
venv/bin/pip install git+https://github.com/Tinkoff/invest-python.git
# затем остальные пакеты: venv/bin/pip install groq aiogram python-dotenv
```

Проверка без токенов (мок-тест логики):
```bash
venv/bin/python dry_run_test.py
```

## 4. Настройка .env

```bash
cp .env.example .env
nano .env
```

Где взять значения:

| Переменная | Где взять |
|---|---|
| `TINKOFF_SANDBOX_TOKEN` | Приложение Т-Инвестиции -> Профиль -> Настройки -> **Токен для песочницы** (нужен подключённый пакет API) |
| `TINKOFF_TOKEN` | Там же -> **Токен для торгового счёта**. Пока тестируете - оставьте пустым |
| `GROQ_API_KEY` | https://console.groq.com/keys - регистрация по email, карта НЕ нужна |
| `TELEGRAM_BOT_TOKEN` | Telegram -> @BotFather -> `/newbot` |
| `TELEGRAM_ADMIN_ID` | Telegram -> @userinfobot -> ваш числовой ID |

```bash
chmod 600 .env   # секреты читает только владелец
```

## 5. Первый запуск (вручную, песочница)

```bash
cd /opt/trading_bot
venv/bin/python main.py
```

- В логе (`logs/bot.log`) увидите: режим ПЕСОЧНИЦА, создание счёта песочницы,
  зачисление 100 000 руб. виртуальных денег.
- В Telegram: `/start` -> статус; `/toggle_trading` -> включить торговлю.
- Останов: Ctrl+C (корректное завершение, `state.json` сохранится).

## 6. Автозапуск через systemd

```bash
sudo cp trading_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading_bot
systemctl status trading_bot          # должно быть active (running)
journalctl -u trading_bot -f          # живые логи
```

Юнит гарантирует: `Restart=always` (подъём через 15 сек после любого падения),
`MemoryMax=400M` (даже при утечке 1 ГБ VPS не умрёт), `KillSignal=SIGINT`
(корректный graceful shutdown).

## 7. Переход на реальный счёт (только когда готовы)

1. Заполните `TINKOFF_TOKEN` в `.env` и `sudo systemctl restart trading_bot`.
2. В Telegram: `/toggle_sandbox` -> предупреждение ->
   `/toggle_sandbox confirm` (действует 5 минут).
3. Бот переключится на реальный счёт и **принудительно выключит торговлю**.
4. `/toggle_trading` - осознанное включение реальной торговли.

## 8. Команды Telegram

| Команда | Действие |
|---|---|
| `/start` | Режим (Sandbox/Real), торговля, аудитор, баланс, аптайм |
| `/status` | Портфель, позиции бота и брокера, последние 3 действия, вердикт последнего аудита |
| `/toggle_trading` | Вкл/выкл автоматическую торговлю |
| `/toggle_sandbox` | Переключение режима (в реал - только с `confirm`) |
| `/toggle_auditor` | Вкл/выкл часового ИИ-аудитора логов |
| `/help` | Справка |

Команды принимает только `TELEGRAM_ADMIN_ID`; остальных бот молча игнорирует.

## 9. Расход лимитов (все - бесплатные)

- **Groq free** (`llama-3.1-8b-instant`): 30 запросов/мин, 14 400/сутки,
  500K токенов/сутки. Бот тратит: сентимент <= 5 раз/6 часов (кэш) + 24 аудита
  в сутки - запас более чем десятикратный. Старая `llama3-8b-8192` Groq
  выведена из эксплуатации, поэтому по умолчанию стоит актуальная модель.
- **T-Invest API**: лимиты 100-300 запросов/мин на метод; бот делает
  ~4 запроса на тикер за цикл (по умолчанию цикл 5 минут) + локальный
  rate-limiter 90 rpm и retry с задержками при `RESOURCE_EXHAUSTED`.

## 10. Диагностика

```bash
tail -n 100 /opt/trading_bot/logs/bot.log    # основной лог
venv/bin/python cleanup.py                   # ручная чистка старых логов
```

| Симптом | Причина/решение |
|---|---|
| `ОШИБКА КОНФИГУРАЦИИ` на старте | Не заполнен .env - смотрите список недостающих переменных в сообщении |
| `SDK Т-Инвестиций не установлен` | Не отработал `pip install -r requirements.txt` (проверьте extra-index-url) |
| Бот молчит в Telegram | Проверьте, что пишете именно с ID = `TELEGRAM_ADMIN_ID` |
| `Тикеры не найдены` | Опечатка в `TRADING_TICKERS` (например, `SBER24` вместо `SBER`) |
| Groq RateLimit в логах | Это нормально: бот ждёт по retry-after и повторяет |

## 11. Как это работает внутри (кратко, для сопровождения)

- `main.py` поднимает в одном asyncio-цикле: long-polling Telegram (основная
  корутина) + `create_task` для движка, аудитора и суточной очистки. SIGINT ->
  все задачи останавливаются, gRPC-канал и HTTP-сессии закрываются.
- `state.json` хранит флаги, открытые позиции и последние действия: после
  падения процесса бот продолжает вести те же позиции (перезапуск systemd
  бесшовный для логики).
- Ордера рыночные, с UUID-ключом идемпотентности `order_id` - сбой сети не
  приводит к двойной покупке.
- Новости: у T-Invest API нет новостного эндпоинта, поэтому `get_news` берёт
  бесплатные RSS (Интерфакс по умолчанию, список настраивается) и фильтрует
  записи по тикеру и названию компании.
