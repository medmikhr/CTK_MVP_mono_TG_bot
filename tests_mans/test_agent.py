#!/usr/bin/env python3
"""
Тестовый скрипт для GigaChat Functions Agent
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def test_agent_initialization():
    """Тест инициализации агента."""
    print("🧪 Тестирование инициализации агента...")
    
    try:
        from class_functions_agent import GigaChatFunctionsAgent
        
        # Проверяем наличие токена
        if not os.getenv('GIGACHAT_TOKEN'):
            print("❌ Не найден токен GigaChat в переменных окружения")
            return False
        
        # Инициализируем агента
        agent = GigaChatFunctionsAgent()
        print("✅ Агент успешно инициализирован")
        
        # Проверяем состояние хранилищ
        store_info = agent.get_store_info()
        print("\n📊 Состояние векторных хранилищ:")
        for store_name, info in store_info.items():
            if "error" not in info:
                status_icon = "✅" if info["status"] == "ready" else "⚠️"
                print(f"   {status_icon} {store_name}: {info['documents']} документов")
            else:
                print(f"   ❌ {store_name}: ошибка - {info['error']}")
        
        # Проверяем функции
        functions_info = agent.get_functions_info()
        print(f"\n🔧 Доступные функции: {functions_info['total_functions']}")
        for func_info in functions_info['functions']:
            print(f"   - {func_info['name']}: {func_info['description'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации агента: {e}")
        return False

def test_function_calls():
    """Тест вызова функций."""
    print("\n🧪 Тестирование вызова функций...")
    
    try:
        from class_functions_agent import GigaChatFunctionsAgent
        
        agent = GigaChatFunctionsAgent()
        
        # Тест функции DAMA поиска
        print("📚 Тестирование DAMA поиска...")
        dama_result = agent.dama_search_func("методология управления данными")
        print(f"   Результат: {len(dama_result.content)} символов")
        print(f"   Источники: {len(dama_result.sources)}")
        
        # Тест функции ЦТК поиска
        print("🔧 Тестирование ЦТК поиска...")
        ctk_result = agent.ctk_search_func("технологические решения")
        print(f"   Результат: {len(ctk_result.content)} символов")
        print(f"   Источники: {len(ctk_result.sources)}")
        
        print("✅ Функции работают корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования функций: {e}")
        return False

def test_agent_queries():
    """Тест запросов к агенту."""
    print("\n🧪 Тестирование запросов к агенту...")
    
    try:
        from class_functions_agent import GigaChatFunctionsAgent
        
        agent = GigaChatFunctionsAgent()
        
        # Тестовые запросы
        test_queries = [
            "Что такое DMBOK?",
            "Расскажи о методологии управления данными",
            "Какие технологические решения предлагает ЦТК?",
            "Сравни подходы DAMA и ЦТК"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Тест {i}: {query}")
            try:
                response = agent.process_query(query)
                print(f"   Ответ: {len(response)} символов")
                print(f"   Начало ответа: {response[:100]}...")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        print("✅ Запросы к агенту обрабатываются")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования запросов: {e}")
        return False

def main():
    """Главная функция тестирования."""
    print("🚀 Запуск тестов GigaChat Functions Agent")
    print("=" * 50)
    
    # Проверяем зависимости
    print("📦 Проверка зависимостей...")
    try:
        import langchain_gigachat
        import langgraph
        import langchain_core
        import pydantic
        print("✅ Все зависимости установлены")
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False
    
    # Запускаем тесты
    tests = [
        test_agent_initialization,
        test_function_calls,
        test_agent_queries
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Результаты тестирования: {passed}/{total} тестов прошли")
    
    if passed == total:
        print("🎉 Все тесты прошли успешно!")
        print("✅ Агент готов к использованию")
        return True
    else:
        print("⚠️ Некоторые тесты не прошли")
        print("Проверьте логи и настройки")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 