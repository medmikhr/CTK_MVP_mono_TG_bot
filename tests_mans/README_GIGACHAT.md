# Data Governance Bot с GigaChat Embeddings

Это улучшенная версия Data Governance бота, которая использует GigaChat embeddings для более качественной обработки и поиска документов.

## Основные улучшения

### 1. GigaChat Embeddings
- Заменены HuggingFace embeddings на GigaChat embeddings
- Улучшенное качество поиска благодаря русскоязычной модели
- Более точное понимание контекста документов

### 2. Улучшенная обработка документов
- Добавлены дополнительные метаданные (дата обработки, модель embeddings)
- Улучшена обработка ошибок
- Более детальное логирование

### 3. Массовая загрузка документов
- Функция загрузки всех документов из папки
- Поддержка рекурсивного обхода подпапок
- Детальная статистика обработки

### 4. Новые файлы
- `document_processor_gigachat.py` - обработчик документов с GigaChat embeddings
- `agent_gigachat.py` - агент с GigaChat embeddings
- `bot_gigachat.py` - бот с GigaChat embeddings
- `bulk_upload.py` - скрипт для массовой загрузки документов

## Структура файлов

```
├── document_processor_gigachat.py  # Обработчик документов с GigaChat
├── agent_gigachat.py              # Агент с GigaChat embeddings
├── bot_gigachat.py                # Telegram бот с GigaChat
├── bulk_upload.py                 # Скрипт массовой загрузки
├── document_processor_langchain.py # Старая версия (HuggingFace)
├── agent.py                       # Старая версия агента
├── bot.py                         # Старая версия бота
├── requirements.txt               # Зависимости
├── INSTALL_WINDOWS.md            # Инструкция для Windows Server
├── INSTALL_LINUX.md              # Инструкция для Linux
└── start_bot.bat                 # Скрипт запуска для Windows
```

## Установка и запуск

### 1. Настройка переменных окружения
Создайте файл `.env`:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GIGACHAT_TOKEN=your_gigachat_token_here
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск бота
Для запуска версии с GigaChat embeddings:
```bash
python bot_gigachat.py
```

Для запуска старой версии:
```bash
python bot.py
```

## Массовая загрузка документов

### Использование скрипта bulk_upload.py

```bash
# Просмотр информации о папке
python bulk_upload.py /path/to/documents dama --info-only

# Предварительный просмотр (без загрузки)
python bulk_upload.py /path/to/documents dama --dry-run

# Загрузка всех документов
python bulk_upload.py /path/to/documents dama

# Загрузка только PDF файлов
python bulk_upload.py /path/to/documents dama --extensions .pdf

# Загрузка с пользовательскими расширениями
python bulk_upload.py /path/to/documents dama --extensions .pdf .docx .txt
```

### Использование в коде

```python
from document_processor_gigachat import process_documents_from_folder, get_folder_info

# Получение информации о папке
folder_info = get_folder_info("/path/to/documents")
print(f"Найдено файлов: {folder_info['total_files']}")

# Обработка документов
result = process_documents_from_folder("/path/to/documents", "dama")
print(f"Обработано: {result['processed']}, Ошибок: {result['failed']}")
```

### Поддерживаемые форматы
- PDF (.pdf)
- Word (.doc, .docx)
- Текстовые файлы (.txt)

## Сравнение версий

| Функция | Старая версия | GigaChat версия |
|---------|---------------|-----------------|
| Embeddings | HuggingFace (rubert-tiny2) | GigaChat |
| Качество поиска | Хорошее | Отличное |
| Русскоязычная поддержка | Ограниченная | Полная |
| Скорость обработки | Быстрая | Средняя |
| Массовая загрузка | Нет | Да |
| Требования к токенам | HF_TOKEN | GIGACHAT_TOKEN |

## Особенности GigaChat версии

### Преимущества:
1. **Лучшее качество поиска** - GigaChat embeddings лучше понимают русский язык
2. **Контекстное понимание** - более точное определение семантической близости
3. **Единая модель** - и LLM, и embeddings используют одну модель GigaChat
4. **Массовая загрузка** - возможность загружать множество документов одновременно

### Ограничения:
1. **Скорость** - GigaChat embeddings работают медленнее HuggingFace
2. **Зависимость от API** - требует стабильного интернет-соединения
3. **Лимиты** - ограничения по количеству запросов к GigaChat API

## Миграция с старой версии

Если у вас уже есть документы, обработанные старой версией:

1. **Новые документы** будут обрабатываться с GigaChat embeddings
2. **Старые документы** останутся доступными для поиска
3. **Рекомендуется** переобработать важные документы с новой версией

## Тестирование

Для тестирования GigaChat embeddings:
```bash
python document_processor_gigachat.py
```

Для тестирования агента:
```bash
python agent_gigachat.py
```

Для тестирования массовой загрузки:
```bash
python bulk_upload.py /path/to/test/documents test_collection --dry-run
```

## Логирование

Все операции логируются с указанием используемой модели embeddings:
- `embedding_model: gigachat` в метаданных документов
- Логи содержат информацию о GigaChat операциях
- Отдельная директория `chroma_db_giga` для хранения данных
- Детальная статистика массовой загрузки

## Поддержка

При возникновении проблем:
1. Проверьте правильность токена GigaChat
2. Убедитесь в стабильности интернет-соединения
3. Проверьте логи на наличие ошибок
4. При необходимости вернитесь к старой версии 