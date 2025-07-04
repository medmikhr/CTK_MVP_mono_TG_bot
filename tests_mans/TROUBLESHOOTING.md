# Руководство по устранению неполадок

## Проблема: Агент вылетает при изменении запроса

### Причины:
1. **500 Internal Server Error** от GigaChat API при использовании инструментов
2. **Нестабильность API** GigaChat
3. **Проблемы с токеном** или сетевым подключением

### Решения:

#### 1. Запустите быстрый тест
```bash
python quick_test.py
```

#### 2. Запустите полную диагностику
```bash
python diagnose_gigachat.py
```

#### 3. Проверьте векторные хранилища
```bash
python check_stores.py
```

#### 4. Тестируйте конкретный инструмент
```bash
# Базовый тест
python test_ctk_tool.py

# Подробный вывод всех событий
python test_ctk_tool.py --verbose
```

#### 5. Используйте простую версию агента
Если диагностика показывает проблемы с инструментами:
```bash
python agent_simple.py
```

#### 6. Используйте обновленную версию с обработкой ошибок
```bash
# Полное тестирование (по умолчанию)
python agent.py

# Быстрый запуск без тестирования инструментов
python agent.py --skip-tests

# Или короткий флаг
python agent.py --fast
```
Теперь агент автоматически переключается на простой режим при ошибках.

#### 7. Проверьте токен GigaChat
```bash
# Проверьте переменную окружения
echo $GIGACHAT_TOKEN  # Linux/Mac
echo %GIGACHAT_TOKEN% # Windows
```

#### 8. Обновите токен
Если токен устарел, получите новый на [GigaChat](https://developers.sber.ru/portal/products/gigachat)

## Типы агентов

### 1. `agent.py` - Полный агент с инструментами
- ✅ Использует векторные хранилища
- ✅ Имеет инструменты для поиска документов
- ⚠️ Может вылетать при проблемах с API

### 2. `agent_simple.py` - Простой агент
- ✅ Стабильная работа
- ✅ Быстрые ответы
- ❌ Без доступа к документам

### 3. `agent_gigachat.py` - Агент с GigaChat embeddings
- ✅ Лучшее качество поиска
- ✅ Улучшенная работа с русским языком
- ⚠️ Требует стабильного API

## Команды для остановки программ

### Корректная остановка:
- Введите: `exit`, `quit`, `выход` или `q`
- Нажмите: `Ctrl+C`

### Принудительная остановка:
- Windows: `Ctrl+Alt+Delete` → Диспетчер задач
- Linux: `kill -9 <PID>`

## Логи и отладка

### Включение подробных логов:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Проверка логов:
- Ищите ошибки `500 Internal Server Error`
- Проверьте время ответа API
- Убедитесь в корректности токена

## Частые проблемы

### 1. "Не найден токен GigaChat"
**Решение:** Проверьте файл `.env` и переменную `GIGACHAT_TOKEN`

### 2. "500 Internal Server Error"
**Решение:** Используйте `agent_simple.py` или подождите и повторите

### 3. "Connection timeout"
**Решение:** Проверьте интернет-соединение и доступность API

### 4. "KeyboardInterrupt"
**Решение:** Это нормально - программа корректно остановлена пользователем

## Рекомендации

1. **Для стабильной работы:** Используйте `agent_simple.py`
2. **Для работы с документами:** Используйте `agent.py` с обработкой ошибок
3. **Для лучшего качества:** Используйте `agent_gigachat.py` при стабильном API
4. **Для диагностики:** Запускайте `diagnose_gigachat.py` при проблемах

## Контакты для поддержки

При постоянных проблемах:
1. Проверьте статус GigaChat API
2. Обновите токен доступа
3. Проверьте сетевое подключение
4. Используйте простую версию агента 