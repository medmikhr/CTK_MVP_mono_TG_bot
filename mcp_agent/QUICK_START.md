# Быстрый старт - GigaChat Functions Agent

## 🚀 Установка и запуск за 5 минут

### 1. Подготовка окружения

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка токена

Создайте файл `.env` в папке `new_agent/`:

```env
GIGACHAT_TOKEN=ваш_токен_gigachat
```

### 3. Загрузка документов (опционально)

Если у вас есть документы DAMA DMBOK и ЦТК:

```bash
# Создайте папки для документов
mkdir -p dama_docs ctk_docs

# Поместите документы в соответствующие папки
# dama_docs/ - документы DAMA DMBOK
# ctk_docs/ - материалы ЦТК

# Запустите загрузку
python load_documents.py dama_docs ctk_docs
```

### 4. Тестирование

```bash
# Запуск тестов
python test_agent.py
```

### 5. Запуск агента

```bash
# Интерактивный режим
python gigachat_functions_agent.py
```

## 📝 Примеры использования

### Программное использование

```python
from gigachat_functions_agent import GigaChatFunctionsAgent

# Инициализация
agent = GigaChatFunctionsAgent()

# Простой запрос
response = agent.process_query("Что такое DMBOK?")
print(response)

# Запрос с памятью (для диалога)
response1 = agent.process_query("Расскажи о методологии управления данными", "user_123")
response2 = agent.process_query("А какие роли есть в этой методологии?", "user_123")
```

### Интерактивный режим

```bash
$ python gigachat_functions_agent.py

🚀 Инициализация GigaChat Functions Agent...

📊 Состояние векторных хранилищ:
   ✅ dama_dmbok: 150 документов
   ✅ ctk_methodology: 75 документов

🔧 Доступные функции: 2
   - dama_search: Поиск информации в документах DAMA DMBOK...
   - ctk_search: Поиск информации в методологических материалах ЦТК...

==================================================
✅ GigaChat Functions Agent готов к работе!
Введите 'exit', 'quit' или 'выход' для завершения
==================================================

Спрашивай: Что такое управление данными?

User: Что такое управление данными?

💬 Bot (за 2.34с):
Управление данными (Data Management) - это комплексная дисциплина...

Спрашивай: Расскажи о ролях в управлении данными

User: Расскажи о ролях в управлении данными

💬 Bot (за 1.87с):
В управлении данными выделяют следующие ключевые роли...
```

## 🔧 Основные функции

### dama_search
Поиск в документах DAMA DMBOK:
- Методологии управления данными
- Стандарты и процессы
- Роли и ответственность
- Data Governance

### ctk_search
Поиск в материалах ЦТК:
- Технологические решения
- Архитектура систем
- Методологии разработки
- Практики и стандарты

## 🐛 Устранение проблем

### Ошибка "Не найден токен GigaChat"
```bash
# Проверьте файл .env
cat .env
# Должно содержать: GIGACHAT_TOKEN=ваш_токен
```

### Ошибка "Пустые векторные хранилища"
```bash
# Загрузите документы
python load_documents.py

# Или проверьте папки
ls -la vector_stores/
```

### Ошибка импорта зависимостей
```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

### Медленная работа
- Первый запуск загружает модели (~100MB)
- Используйте `device='cpu'` для экономии памяти
- Уменьшите `k` в `similarity_search` для ускорения

## 📊 Мониторинг

### Логи
Агент ведет подробные логи:
```bash
# Просмотр логов в реальном времени
tail -f agent.log
```

### Состояние хранилищ
```python
agent = GigaChatFunctionsAgent()
store_info = agent.get_store_info()
print(store_info)
```

### Информация о функциях
```python
functions_info = agent.get_functions_info()
print(functions_info)
```

## 🔄 Обновления

### Добавление новой функции
```python
@giga_tool(few_shot_examples=examples)
def new_search(query: str) -> SearchResult:
    """Описание функции"""
    # Логика поиска
    return SearchResult(content="...", sources=[])
```

### Изменение модели embeddings
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="другая_модель",
    model_kwargs={'device': 'cpu'}
)
```

## 📞 Поддержка

При проблемах:
1. Запустите `python test_agent.py`
2. Проверьте логи агента
3. Убедитесь в корректности токена
4. Проверьте наличие документов

## 🎯 Следующие шаги

1. **Загрузите документы** - для полноценной работы
2. **Настройте логирование** - для отладки
3. **Добавьте функции** - для расширения возможностей
4. **Интегрируйте в систему** - для production использования

## Структура директорий

```
new_agent/
├── agent.py              # Основной агент
├── load_documents.py     # Загрузка документов
├── test_agent.py         # Тестирование
├── requirements.txt      # Зависимости
├── README.md            # Полная документация
├── QUICK_START.md       # Этот файл
├── .env                 # Переменные окружения
└── vector_stores/       # Векторные хранилища (создается автоматически)

../dama_docs/            # Документы DAMA DMBOK
../ctk_docs/             # Материалы ЦТК
```