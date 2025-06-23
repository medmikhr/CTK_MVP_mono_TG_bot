# Changelog - GigaChat Functions Agent

## [2.0.0] - 2024-12-19

### 🚀 Major Changes - GigaChat Function Calling Implementation

Полная переработка агента в соответствии с официальными рекомендациями Сбера для работы с функциями GigaChat.

#### ✨ Новые возможности

- **GigaChat Function Calling** - использование официального подхода Сбера
- **giga_tool декоратор** - для создания функций с few-shot примерами
- **create_react_agent** - агент с памятью и поддержкой функций
- **Структурированные ответы** - с использованием Pydantic моделей
- **Память диалога** - сохранение контекста между запросами
- **Few-shot примеры** - для улучшения понимания функций моделью

#### 🔧 Технические изменения

##### Архитектура
- Заменен кастомный function calling на `giga_tool` декоратор
- Интеграция с LangGraph для создания агента
- Использование `bind_functions` для привязки функций к LLM
- Добавлен `MemorySaver` для сохранения контекста

##### Функции
```python
# Старый подход
def dama_search(self, query: str) -> str:
    # Логика поиска
    return "результат"

# Новый подход
@giga_tool(few_shot_examples=examples)
def dama_search(query: str = Field(description="...")) -> DamaSearchResult:
    # Логика поиска
    return DamaSearchResult(content="...", sources=[])
```

##### Структурированные ответы
```python
class DamaSearchResult(BaseModel):
    content: str = Field(description="Найденная информация")
    sources: List[str] = Field(description="Список источников")
```

#### 📦 Зависимости

**Добавлены:**
- `langgraph>=0.1.0` - для создания агента
- `langchain-core>=0.1.0` - для базовых компонентов
- `pydantic>=2.0.0` - для валидации данных

**Обновлены:**
- `langchain-gigachat>=0.1.0` - для поддержки giga_tool
- `langchain-chroma>=0.1.0` - для векторного поиска

#### 🔄 API Changes

##### Инициализация
```python
# Старый код
agent = DataManagementAgent()
response = agent.process_query_simple(query)

# Новый код
agent = GigaChatFunctionsAgent()
response = agent.process_query(query, thread_id="user_123")
```

##### Обработка запросов
- Добавлена поддержка thread_id для памяти диалога
- Автоматический вызов функций GigaChat
- Fallback к простому LLM при ошибках

#### 📚 Документация

- Обновлен README.md с описанием нового подхода
- Создан QUICK_START.md для быстрого старта
- Добавлен test_agent.py для тестирования
- Создан CHANGELOG.md для отслеживания изменений

#### 🧪 Тестирование

- Добавлены тесты инициализации агента
- Тесты вызова функций
- Тесты обработки запросов
- Проверка зависимостей

#### 🐛 Исправления

- Устранены проблемы с совместимостью GigaChat API
- Улучшена обработка ошибок
- Добавлен graceful degradation
- Оптимизирована производительность

#### 📈 Улучшения производительности

- Кэширование embeddings
- Оптимизированные запросы к векторному хранилищу
- Асинхронная обработка где возможно
- Улучшенное логирование

---

## [1.0.0] - 2024-12-18

### 🎉 Initial Release

Первая версия агента с базовой функциональностью:
- Поиск в документах DAMA DMBOK
- Поиск в материалах ЦТК
- Векторный поиск с ChromaDB
- Локальные embeddings на русском языке
- Базовая интеграция с GigaChat

---

## Ключевые отличия версии 2.0.0

### ✅ Соответствие рекомендациям Сбера

1. **giga_tool декоратор** - официальный способ создания функций
2. **bind_functions** - правильная привязка функций к LLM
3. **create_react_agent** - рекомендуемый агент с функциями
4. **Few-shot примеры** - для улучшения понимания функций
5. **Структурированные ответы** - с использованием Pydantic

### 🔄 Миграция с версии 1.0.0

```python
# Старый код
from agent import DataManagementAgent
agent = DataManagementAgent()
response = agent.process_query_simple("запрос")

# Новый код
from gigachat_functions_agent import GigaChatFunctionsAgent
agent = GigaChatFunctionsAgent()
response = agent.process_query("запрос", "thread_id")
```

### 📊 Преимущества новой версии

- **Лучшая совместимость** с GigaChat API
- **Автоматический вызов функций** моделью
- **Память диалога** для контекстных разговоров
- **Структурированные ответы** для лучшей обработки
- **Few-shot примеры** для улучшения качества
- **Graceful degradation** при ошибках 