#!/usr/bin/env python3
"""
Скрипт для тестирования ctk_retrieve_tool
"""

import time
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

def test_ctk_tool_direct():
    """Тестирование ctk_retrieve_tool напрямую."""
    print("🔍 Тестирование ctk_retrieve_tool напрямую")
    print("=" * 50)
    
    try:
        # Импортируем реальные компоненты
        from old_react_agent import ctk_retrieve_tool, vector_stores
        
        # Проверяем состояние хранилища
        ctk_store = vector_stores["ctk"]
        count = ctk_store._collection.count()
        print(f"Хранилище CTK: {count} документов")
        
        if count == 0:
            print("❌ Хранилище CTK пустое")
            return False
        
        # Тестируем инструмент
        test_queries = [
            "слои информационной архитектуры",
            "технологические решения",
            "архитектура систем"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Тест #{i}: '{query}'")
            print("-" * 30)
            
            start_time = time.time()
            
            try:
                result = ctk_retrieve_tool.invoke(query)
                end_time = time.time()
                
                print(f"✅ Результат получен за {end_time - start_time:.2f}с")
                print(f"📄 Тип результата: {type(result).__name__}")
                
                # Детальный анализ результата
                if hasattr(result, 'content'):
                    print(f"📝 Контент: {result.content[:300]}...")
                elif isinstance(result, str):
                    print(f"📝 Строка: {result[:300]}...")
                elif isinstance(result, tuple):
                    print(f"📝 Кортеж: {len(result)} элементов")
                    for j, item in enumerate(result):
                        print(f"   Элемент {j+1}: {str(item)[:100]}...")
                else:
                    print(f"📝 Объект: {str(result)[:300]}...")
                
                # Проверяем дополнительные атрибуты
                if hasattr(result, 'metadata'):
                    print(f"🏷️  Метаданные: {result.metadata}")
                if hasattr(result, 'additional_kwargs'):
                    additional = result.additional_kwargs
                    if additional:
                        print(f"🔧 Дополнительно: {additional}")
                
            except Exception as e:
                end_time = time.time()
                print(f"❌ Ошибка за {end_time - start_time:.2f}с: {e}")
                print(f"   Тип ошибки: {type(e).__name__}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        return False

def test_ctk_tool_via_agent():
    """Тестирование ctk_retrieve_tool через агента."""
    print("\n🔍 Тестирование ctk_retrieve_tool через агента")
    print("=" * 50)
    
    try:
        from old_react_agent import agent_executor
        
        test_queries = [
            "Используй ctk_retrieve_tool для поиска информации о слоях информационной архитектуры",
            "Найди информацию о технологических решениях используя ctk_retrieve_tool"
        ]
        
        for query in test_queries:
            print(f"\nТестирование запроса: '{query}'")
            print("-" * 50)
            
            start_time = time.time()
            config = {"configurable": {"thread_id": "test_ctk"}}
            
            try:
                event_count = 0
                for event in agent_executor.stream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="values",
                    config=config,
                ):
                    event_count += 1
                    print(f"\n📋 Событие #{event_count}:")
                    
                    # Выводим тип события
                    if "messages" in event:
                        messages = event["messages"]
                        if messages:
                            last_message = messages[-1]
                            print(f"   Тип: {type(last_message).__name__}")
                            
                            # Если это сообщение от агента
                            if hasattr(last_message, 'content'):
                                print(f"   Контент: {last_message.content[:200]}...")
                            
                            # Если есть дополнительные атрибуты
                            if hasattr(last_message, 'additional_kwargs'):
                                additional = last_message.additional_kwargs
                                if additional:
                                    print(f"   Дополнительно: {additional}")
                    
                    # Выводим все ключи события для отладки
                    print(f"   Ключи события: {list(event.keys())}")
                    
                    # Если есть tool_calls
                    if "tool_calls" in event:
                        tool_calls = event["tool_calls"]
                        print(f"   🔧 Вызовы инструментов: {len(tool_calls)}")
                        for i, tool_call in enumerate(tool_calls):
                            print(f"      Инструмент {i+1}: {tool_call}")
                    
                    # Если есть tool_results
                    if "tool_results" in event:
                        tool_results = event["tool_results"]
                        print(f"   📊 Результаты инструментов: {len(tool_results)}")
                        for i, result in enumerate(tool_results):
                            print(f"      Результат {i+1}: {str(result)[:200]}...")
                
                end_time = time.time()
                print(f"\n✅ Всего событий: {event_count}")
                print(f"Время ответа: {end_time - start_time:.2f}с")
                
            except Exception as e:
                print(f"❌ Ошибка при запросе: {e}")
                print(f"Тип ошибки: {type(e).__name__}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Главная функция тестирования."""
    import sys
    
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("🚀 Тестирование ctk_retrieve_tool")
    if verbose:
        print("📋 Режим подробного вывода включен")
    print("=" * 50)
    
    # Тест напрямую
    print("\n1️⃣ Тестирование прямого вызова...")
    direct_success = test_ctk_tool_direct()
    
    # Тест через агента
    print("\n2️⃣ Тестирование через агента...")
    agent_success = test_ctk_tool_via_agent()
    
    # Итоговый отчет
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    print(f"Прямой вызов ctk_retrieve_tool: {'✅' if direct_success else '❌'}")
    print(f"Вызов через агента: {'✅' if agent_success else '❌'}")
    
    if direct_success and agent_success:
        print("\n✅ ctk_retrieve_tool работает корректно!")
        print("   Все тесты пройдены успешно")
    elif direct_success and not agent_success:
        print("\n⚠️  ctk_retrieve_tool работает напрямую, но есть проблемы с агентом")
        print("   Возможные причины:")
        print("   - Нестабильность GigaChat API при сложных запросах")
        print("   - Проблемы с LangGraph агентом")
        print("   - Ошибки в обработке инструментов")
    elif not direct_success:
        print("\n❌ Проблемы с ctk_retrieve_tool")
        print("Проверьте:")
        print("1. Загружены ли документы в хранилище")
        print("2. Работают ли embeddings")
        print("3. Состояние ChromaDB")
        print("4. Доступность файлов в PERSIST_DIR")
    
    print(f"\n💡 Для более подробного вывода используйте: python test_ctk_tool.py --verbose")

if __name__ == "__main__":
    main() 