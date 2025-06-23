#!/usr/bin/env python3
"""
Упрощенный агент с прямым использованием инструментов
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_chroma import Chroma
from langchain.agents import tool
from document_processor import PERSIST_DIR
from embeddings_manager import get_local_huggingface_embeddings
import time
import sys

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение токенов из переменных окружения
GC_AUTH = os.getenv('GIGACHAT_TOKEN')
if not GC_AUTH:
    raise ValueError("Не найден токен GigaChat в переменных окружения")

# Инициализация GigaChat
llm = GigaChat(
    credentials=GC_AUTH,
    model='GigaChat:latest',
    verify_ssl_certs=False,
    profanity_check=False
)

# Получение embeddings из общего менеджера
embeddings = get_local_huggingface_embeddings()

# Инициализация векторных хранилищ
vector_stores = {
    "dama": Chroma(collection_name="dama", persist_directory=PERSIST_DIR, embedding_function=embeddings),
    "ctk": Chroma(collection_name="ctk", persist_directory=PERSIST_DIR, embedding_function=embeddings),
    "sbf": Chroma(collection_name="sbf", persist_directory=PERSIST_DIR, embedding_function=embeddings)
}

def search_documents(store: Chroma, query: str, k: int = 5) -> str:
    """Общая функция для поиска документов в векторном хранилище."""
    try:
        retrieved_docs = store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {e}")
        return f"Ошибка при поиске: {str(e)}"

@tool
def dama_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о методологии управления данными, 
    стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными.
    Этот инструмент содержит информацию из Data Management Body Of Knowledge (DMBOK)."""
    return search_documents(vector_stores["dama"], query)

@tool
def ctk_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о технологических решениях, 
    архитектуре систем, методологиях разработки, стандартах и практиках ЦТК.
    Этот инструмент содержит документацию Центра Технологического Консалтинга."""
    return search_documents(vector_stores["ctk"], query)

@tool
def sbf_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о факторинговых операциях, 
    продуктах и услугах СберБанк Факторинга, процессах работы с клиентами.
    Этот инструмент содержит документацию СберБанк Факторинга."""
    return search_documents(vector_stores["sbf"], query)

def simple_agent_ask(user_input: str) -> str:
    """Упрощенная функция агента с прямым использованием инструментов."""
    try:
        print(f"\n🔍 Обработка запроса: '{user_input}'")
        print("=" * 50)
        
        # Определяем, какой инструмент использовать на основе ключевых слов
        tools_to_use = []
        
        # Проверяем ключевые слова для каждого инструмента
        dama_keywords = ['dama', 'управление данными', 'методология', 'стандарты', 'dmbok']
        ctk_keywords = ['ctk', 'технологии', 'архитектура', 'разработка', 'системы']
        sbf_keywords = ['sbf', 'факторинг', 'сбербанк факторинг', 'финансы']
        
        user_input_lower = user_input.lower()
        
        if any(keyword in user_input_lower for keyword in dama_keywords):
            tools_to_use.append(("dama_retrieve_tool", dama_retrieve_tool))
        
        if any(keyword in user_input_lower for keyword in ctk_keywords):
            tools_to_use.append(("ctk_retrieve_tool", ctk_retrieve_tool))
        
        if any(keyword in user_input_lower for keyword in sbf_keywords):
            tools_to_use.append(("sbf_retrieve_tool", sbf_retrieve_tool))
        
        # Если не найдены ключевые слова, используем все инструменты
        if not tools_to_use:
            tools_to_use = [
                ("dama_retrieve_tool", dama_retrieve_tool),
                ("ctk_retrieve_tool", ctk_retrieve_tool),
                ("sbf_retrieve_tool", sbf_retrieve_tool)
            ]
        
        # Собираем информацию из всех подходящих инструментов
        collected_info = []
        
        for tool_name, tool_func in tools_to_use:
            print(f"\n🔧 Используем {tool_name}...")
            try:
                result = tool_func.invoke(user_input)
                if result and len(result.strip()) > 0:
                    collected_info.append(f"=== Информация из {tool_name} ===\n{result}")
                    print(f"✅ Получено {len(result)} символов")
                else:
                    print(f"⚠️  Пустой результат от {tool_name}")
            except Exception as e:
                print(f"❌ Ошибка {tool_name}: {e}")
        
        if not collected_info:
            print("⚠️  Не удалось получить информацию из инструментов")
            # Пробуем простой запрос к LLM
            response = llm.invoke(user_input)
            return response.content
        
        # Формируем контекст для LLM
        context = "\n\n".join(collected_info)
        
        # Создаем промпт с контекстом
        prompt = f"""На основе предоставленной информации ответь на вопрос пользователя.

Контекст:
{context}

Вопрос пользователя: {user_input}

Ответь подробно и структурированно, используя информацию из контекста. Если в контексте нет информации для ответа, скажи об этом честно."""
        
        print(f"\n🤖 Отправляем запрос к LLM...")
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Ошибка в simple_agent_ask: {e}")
        
        # Fallback к простому запросу
        try:
            print("🔄 Попытка простого запроса...")
            response = llm.invoke(user_input)
            return response.content
        except Exception as fallback_error:
            logger.error(f"Ошибка fallback: {fallback_error}")
            return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"

def test_simple_agent():
    """Тестирование упрощенного агента."""
    print("🧪 Тестирование упрощенного агента...")
    print("=" * 50)
    
    test_queries = [
        "Что такое управление данными?",
        "Расскажи о технологических решениях",
        "Что такое факторинг?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Тест #{i}: '{query}'")
        print("-" * 30)
        
        start_time = time.time()
        try:
            result = simple_agent_ask(query)
            end_time = time.time()
            
            print(f"✅ Ответ получен за {end_time - start_time:.2f}с")
            print(f"📝 Длина ответа: {len(result)} символов")
            print(f"📄 Ответ: {result[:300]}...")
            
        except Exception as e:
            end_time = time.time()
            print(f"❌ Ошибка за {end_time - start_time:.2f}с: {e}")

if __name__ == '__main__':
    # Проверяем флаги запуска
    run_tests = "--test" in sys.argv
    
    if run_tests:
        test_simple_agent()
        exit(0)
    
    print("🚀 Упрощенный агент запущен!")
    print("Введите 'exit', 'quit' или 'выход' для завершения")
    print("Или нажмите Ctrl+C для принудительной остановки")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("Спрашивай: ")
            
            # Проверка команд выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                print("Выход по команде пользователя")
                break
            
            result = simple_agent_ask(user_input)
            print(f"\n💬 Ответ:\n{result}")
            
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем (Ctrl+C)")
        print("До свидания!")
    except EOFError:
        print("\n\nПрограмма завершена (EOF)")
        print("До свидания!") 