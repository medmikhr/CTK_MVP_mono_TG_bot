# Инструкция по установке бота на Linux сервер

## 1. Подготовка окружения

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## 2. Настройка переменных окружения

Создайте файл `.env` в корневой директории проекта со следующим содержимым:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GIGACHAT_TOKEN=your_gigachat_token_here
HF_TOKEN=your_huggingface_token_here
```

## 3. Настройка systemd сервиса

1. Создайте файл сервиса:

```bash
sudo nano /etc/systemd/system/ctk_bot.service
```

2. Добавьте следующее содержимое (замените пути на актуальные):

```ini
[Unit]
Description=CTK Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/your/bot
Environment=PYTHONPATH=/path/to/your/bot
ExecStart=/path/to/your/bot/.venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Активируйте и запустите сервис:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ctk_bot
sudo systemctl start ctk_bot
```

4. Проверьте статус:

```bash
sudo systemctl status ctk_bot
```

## 4. Запуск в screen (альтернативный вариант)

Если нет возможности использовать systemd, можно использовать screen:

```bash
# Установка screen
sudo apt-get install screen

# Создание нового screen сессии
screen -S ctk_bot

# Активация виртуального окружения и запуск бота
source .venv/bin/activate
python bot.py

# Отключение от screen сессии (бот продолжит работать)
# Нажмите Ctrl+A, затем D

# Чтобы вернуться к сессии:
screen -r ctk_bot
```

## 5. Логирование

Логи бота можно просмотреть следующими способами:

- При использовании systemd:
```bash
journalctl -u ctk_bot -f
```

- При использовании screen:
```bash
# Подключитесь к screen сессии
screen -r ctk_bot
```

## 6. Обновление бота

1. Остановите сервис:
```bash
# Для systemd:
sudo systemctl stop ctk_bot

# Для screen:
screen -X -S ctk_bot quit
```

2. Обновите код
3. Обновите зависимости:
```bash
pip install -r requirements.txt
```

4. Перезапустите сервис:
```bash
# Для systemd:
sudo systemctl start ctk_bot

# Для screen:
screen -S ctk_bot
source .venv/bin/activate
python bot.py
``` 