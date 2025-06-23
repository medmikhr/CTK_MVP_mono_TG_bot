#!/usr/bin/env python3
"""
Простой тест инструментов
"""

import os
import time
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

def test_simple_tool_calls():
    """Тестирование простых вызовов инструментов."""
    print("🔍 Простой тест инструментов")
    print("=" * 50)
    
    try:
        from old_react_agent import dama_retrieve_tool, ctk_retrieve_tool, sbf_retrieve_tool
        
        # Простые тестовые запросы
        test_cases = [
            ("dama_retrieve_tool", "управление данными", dama_retrieve_tool),
            ("ctk_retrieve_tool", "архитектура", ctk_retrieve_tool),
            ("sbf_retrieve_tool", "факторинг", sbf_retrieve_tool),
        ]
        
        for tool_name, query, tool_func in test_cases:
            print(f"\n🔧 Тест {tool_name}: '{query}'")
            print("-" * 30)
            
            start_time = time.time()
            
            try:
                result = tool_func.invoke(query)
                end_time = time.time()
                
                print(f"✅ Результат получен за {end_time - start_time:.2f}с")
                print(f"📄 Тип: {type(result).__name__}")
                
                if isinstance(result, str):
                    print(f"📝 Длина: {len(result)} символов")
                    print(f"📄 Начало: {result[:200]}...")
                else:
                    print(f"📝 Результат: {str(result)[:200]}...")
                    
            except Exception as e:
                end_time = time.time()
                print(f"❌ Ошибка за {end_time - start_time:.2f}с: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False

def test_agent_with_simple_query():
    """Тестирование агента с простым запросом."""
    print("\n🤖 Тест агента с простым запросом")
    print("=" * 50)
    
    try:
        from old_react_agent import agent_executor
        
        # Простой запрос
        simple_query = "Что такое управление данными?"
        
        print(f"Запрос: '{simple_query}'")
        print("-" * 30)
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "simple_test"}}
        
        event_count = 0
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": simple_query}]},
            stream_mode="values",
            config=config,
        ):
            event_count += 1
            print(f"\n📋 Событие #{event_count}:")
            
            if "messages" in event and event["messages"]:
                messages = event["messages"]
                for i, message in enumerate(messages):
                    if hasattr(message, 'content'):
                        print(f"   Сообщение {i+1}: {message.content[:200]}...")
                        
                        # Проверяем вызовы инструментов
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            print(f"   🔧 Вызовы инструментов: {len(message.tool_calls)}")
                            for tool_call in message.tool_calls:
                                print(f"      - {tool_call}")
        
        end_time = time.time()
        print(f"\n✅ Завершено за {end_time - start_time:.2f}с ({event_count} событий)")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Главная функция."""
    print("🚀 Простой тест инструментов")
    print("=" * 50)
    
    # Тест прямых вызовов
    print("\n1️⃣ Тест прямых вызовов инструментов...")
    direct_success = test_simple_tool_calls()
    
    # Тест агента
    print("\n2️⃣ Тест агента...")
    agent_success = test_agent_with_simple_query()
    
    # Итоги
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 50)
    
    print(f"Прямые вызовы: {'✅' if direct_success else '❌'}")
    print(f"Агент: {'✅' if agent_success else '❌'}")
    
    if direct_success and agent_success:
        print("\n✅ Все тесты пройдены!")
    else:
        print("\n⚠️  Есть проблемы с инструментами")

if __name__ == "__main__":
    main() 