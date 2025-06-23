#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы GigaChat Tool Calling Agent
"""

from gigachat_tool_calling_agent import call_agent

def test_queries():
    """Тестируем различные запросы."""
    
    test_queries = [
        "Что такое DAMA DMBOK?",
        "Расскажи о методологии ЦТК",
        "Привет, как дела?"  # Общий запрос без ключевых слов
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ {i}: {query}")
        print(f"{'='*60}")
        
        try:
            response = call_agent(query, thread_id=f"test_{i}")
            print(f"\nОТВЕТ:")
            print(response)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    print("🧪 Тестирование GigaChat Tool Calling Agent")
    test_queries() 