# Инструкция по установке бота на Windows Server

## 1. Установка Python

1. Скачайте и установите Python 3.10 или выше с официального сайта: https://www.python.org/downloads/windows/
2. При установке обязательно отметьте галочку "Add Python to PATH"
3. Проверьте установку, открыв PowerShell и выполнив команду:
```powershell
python --version
```

## 2. Подготовка окружения

1. Откройте PowerShell от имени администратора и перейдите в директорию проекта:
```powershell
cd C:\path\to\your\bot
```

2. Создайте и активируйте виртуальное окружение:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Установите зависимости:
```powershell
pip install -r requirements.txt
```

## 3. Настройка переменных окружения

1. Создайте файл `.env` в корневой директории проекта со следующим содержимым:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GIGACHAT_TOKEN=your_gigachat_token_here
HF_TOKEN=your_huggingface_token_here
```

## 4. Создание Windows Service

1. Установите nssm (Non-Sucking Service Manager):
   - Скачайте nssm с https://nssm.cc/download
   - Распакуйте архив
   - Скопируйте nssm.exe (из папки win64) в C:\Windows\System32

2. Откройте PowerShell от имени администратора и создайте службу:
```powershell
# Перейдите в директорию проекта
cd C:\path\to\your\bot

# Создайте службу
nssm install CTKBot "C:\path\to\your\bot\.venv\Scripts\python.exe" "C:\path\to\your\bot\bot.py"

# Настройте рабочую директорию
nssm set CTKBot AppDirectory "C:\path\to\your\bot"

# Настройте переменные окружения
nssm set CTKBot AppEnvironmentExtra PATH=C:\path\to\your\bot\.venv\Scripts;%PATH%

# Настройте автоматический перезапуск при сбоях
nssm set CTKBot AppRestartDelay 10000
```

3. Запустите службу:
```powershell
Start-Service CTKBot
```

4. Проверьте статус службы:
```powershell
Get-Service CTKBot
```

## 5. Альтернативный вариант запуска (без службы)

Если вы не хотите использовать Windows Service, можно создать bat-файл для запуска:

1. Создайте файл `start_bot.bat`:
```batch
@echo off
cd /d %~dp0
call .venv\Scripts\activate
python bot.py
pause
```

2. Создайте ярлык для этого файла (правый клик -> Создать ярлык)

3. Добавьте ярлык в автозагрузку одним из способов:

   Для всех пользователей (рекомендуется для сервера):
   - Скопируйте ярлык в `C:\ProgramData\Microsoft\Windows\Start Menu\Programs\StartUp`
   - Или используйте команду `shell:common startup` в Win+R

   Для текущего пользователя:
   - Скопируйте ярлык в `C:\Users\[Username]\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
   - Или используйте команду `shell:startup` в Win+R

## 6. Логирование

Логи бота можно просмотреть следующими способами:

- При использовании Windows Service:
```powershell
# Просмотр логов службы
Get-EventLog -LogName Application -Source CTKBot -Newest 50

# Или через Event Viewer (eventvwr.msc)
```

- При использовании bat-файла:
  - Логи будут выводиться в консоль
  - Можно добавить перенаправление в файл в bat-файл:
```batch
python bot.py >> bot.log 2>&1
```

## 7. Обновление бота

1. Остановите службу:
```powershell
# Если используется служба
Stop-Service CTKBot

# Если используется bat-файл
# Закройте окно с ботом
```

2. Обновите код

3. Обновите зависимости:
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

4. Запустите службу:
```powershell
# Если используется служба
Start-Service CTKBot

# Если используется bat-файл
# Запустите start_bot.bat
```

## Решение проблем

1. Если служба не запускается:
   - Проверьте логи в Event Viewer
   - Убедитесь, что все пути в настройках службы указаны правильно
   - Проверьте права доступа к директории проекта

2. Если бот не отвечает:
   - Проверьте подключение к интернету
   - Проверьте правильность токенов в файле .env
   - Проверьте статус службы
   - Проверьте логи 