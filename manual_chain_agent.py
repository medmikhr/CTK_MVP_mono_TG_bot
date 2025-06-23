#!/usr/bin/env python3
"""
Упрощенный агент с прямым использованием инструментов и встроенной памятью LangChain
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain.agents import tool
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    ConversationBufferWindowMemory
)
from langchain_core.messages import HumanMessage, AIMessage
from document_processor import search_documents
import time
import sys
import inspect

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

# Словарь для хранения памяти пользователей
user_memories = {}

# Тип памяти по умолчанию
DEFAULT_MEMORY_TYPE = "buffer"  # buffer, summary, token_buffer, window

@tool
def dama_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о методологии управления данными, 
    стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными.
    Этот инструмент содержит информацию из Data Management Body Of Knowledge (DMBOK)."""
    results = search_documents(query, "dama_dmbok", n_results=5)
    if not results:
        return "Информация не найдена в стандарте DAMA DMBOK."
    
    content_parts = []
    for i, result in enumerate(results, 1):
        source = result['metadata'].get('source', 'Неизвестный источник')
        score = result['score']
        clean_text = result['text'].replace('\n', ' ').replace('  ', ' ').strip()
        content_parts.append(f"\nИсточник {i}: {source} (релевантность: {score:.3f})\n{clean_text}")
    
    return "\n\n---\n".join(content_parts)

@tool
def ctk_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о технологических решениях, 
    архитектуре систем, методологиях разработки, стандартах и практиках ЦТК.
    Этот инструмент содержит документацию Центра Технологического Консалтинга."""
    results = search_documents(query, "ctk_methodology", n_results=5)
    if not results:
        return "Информация не найдена в регламентах и методологических материалах ЦТК."
    
    content_parts = []
    for i, result in enumerate(results, 1):
        source = result['metadata'].get('source', 'Неизвестный источник')
        score = result['score']
        clean_text = result['text'].replace('\n', ' ').replace('  ', ' ').strip()
        content_parts.append(f"\nИсточник {i}: {source} (релевантность: {score:.3f})\n{clean_text}")
    
    return "\n\n---\n".join(content_parts)

@tool
def sbf_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации в искусственных данных и метаданных, 
    созданных для демонстрационных целей СБФ. Эти данные не имеют отношения к реальной деятельности компании.
    Этот инструмент содержит синтезированные метаданные для СБФ."""
    results = search_documents(query, "sbf_meta", n_results=5)
    if not results:
        return "Информация не найдена в синтезированных метаданных компании СберФакторинг (СБФ)."
    
    content_parts = []
    for i, result in enumerate(results, 1):
        source = result['metadata'].get('source', 'Неизвестный источник')
        score = result['score']
        clean_text = result['text'].replace('\n', ' ').replace('  ', ' ').strip()
        content_parts.append(f"\nИсточник {i}: {source} (релевантность: {score:.3f})\n{clean_text}")
    
    return "\n\n---\n".join(content_parts)

def get_functions_info():
    """Получение информации о доступных инструментах для бота."""
    # Собираем все функции с декоратором @tool
    tool_functions = []
    
    # Получаем все функции из текущего модуля
    current_module = inspect.currentframe().f_globals
    
    for name, obj in current_module.items():
        if inspect.isfunction(obj) and hasattr(obj, '__wrapped__') and hasattr(obj, 'name'):
            # Это функция с декоратором @tool
            tool_functions.append({
                "name": obj.name,
                "description": obj.__doc__ or "Описание отсутствует",
                "function": obj
            })
    
    return {
        "total_functions": len(tool_functions),
        "function_names": [func["name"] for func in tool_functions],
        "functions": [
            {
                "name": func["name"],
                "description": func["description"],
                "args_schema": getattr(func["function"], 'args_schema', None)
            }
            for func in tool_functions
        ]
    }

def create_memory(memory_type: str = DEFAULT_MEMORY_TYPE, thread_id: str = "default"):
    """Создает память указанного типа."""
    if memory_type == "buffer":
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
    elif memory_type == "summary":
        return ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
    elif memory_type == "token_buffer":
        return ConversationTokenBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
    elif memory_type == "window":
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Хранит последние 5 пар сообщений
        )
    else:
        # По умолчанию используем buffer
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )

def get_user_memory(thread_id: str, memory_type: str = DEFAULT_MEMORY_TYPE):
    """Получает или создает память для пользователя."""
    memory_key = f"{thread_id}_{memory_type}"
    if memory_key not in user_memories:
        user_memories[memory_key] = create_memory(memory_type, thread_id)
    return user_memories[memory_key]

def call_agent(user_input: str, thread_id: str = "default", memory_type: str = DEFAULT_MEMORY_TYPE) -> str:
    """Упрощенная функция агента с прямым использованием инструментов и встроенной памятью LangChain."""
    try:
        print(f"\n🔍 Обработка запроса: '{user_input}'")
        print(f"🧠 Тип памяти: {memory_type}")
        print("=" * 50)
        
        # Получаем память пользователя
        memory = get_user_memory(thread_id, memory_type)
        
        # Определяем, какой инструмент использовать на основе ключевых слов
        tools_to_use = []
        
        # Проверяем ключевые слова для каждого инструмента
        dama_keywords = ['dama', 'управление данными', 'методология', 'стандарты', 'dmbok']
        ctk_keywords = ['ctk', 'регламенты', 'архитектура', 'ролевая модель', 'характеристики качества', 'ЦТК']
        sbf_keywords = ['sbf', 'сберфакторинг', 'сбербанк факторинг', 'метаданные сбф', 'СБФ', 'искусственные данные']
        
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
            # Пробуем простой запрос к LLM с памятью
            messages = memory.chat_memory.messages + [HumanMessage(content=user_input)]
            response = llm.invoke(messages)
            bot_response = response.content
        else:
            # Формируем контекст для LLM с памятью
            context = "\n\n".join(collected_info)
            
            # Получаем историю из памяти
            chat_history = memory.chat_memory.messages
            
            # Создаем промпт с контекстом и историей
            history_context = ""
            if chat_history:
                # Добавляем последние 3 пары вопрос-ответ для контекста
                recent_history = chat_history[-6:]  # Последние 6 сообщений (3 пары)
                history_parts = []
                for i in range(0, len(recent_history), 2):
                    if i + 1 < len(recent_history):
                        history_parts.append(f"Пользователь: {recent_history[i].content}")
                        history_parts.append(f"Ассистент: {recent_history[i+1].content}")
                if history_parts:
                    history_context = "\n\nИстория диалога:\n" + "\n".join(history_parts) + "\n\n"
            
            prompt = f"""На основе предоставленной информации и истории диалога ответь на вопрос пользователя.

{history_context}Контекст:
{context}

Текущий вопрос пользователя: {user_input}

Ответь подробно и структурированно, используя информацию из контекста. Если в контексте нет информации для ответа, скажи об этом честно. Учитывай историю диалога для более точного ответа."""
            
            print(f"\n🤖 Отправляем запрос к LLM...")
            response = llm.invoke(prompt)
            bot_response = response.content
        
        # Сохраняем сообщения в память
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(bot_response)
        
        return bot_response
        
    except Exception as e:
        logger.error(f"Ошибка в call_agent: {e}")
        
        # Fallback к простому запросу с памятью
        try:
            print("🔄 Попытка простого запроса...")
            memory = get_user_memory(thread_id, memory_type)
            messages = memory.chat_memory.messages + [HumanMessage(content=user_input)]
            response = llm.invoke(messages)
            bot_response = response.content
            
            # Сохраняем в память
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(bot_response)
            
            return bot_response
        except Exception as fallback_error:
            logger.error(f"Ошибка fallback: {fallback_error}")
            return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"

def clear_conversation_history(thread_id: str = "default", memory_type: str = DEFAULT_MEMORY_TYPE):
    """Очищает историю диалога для указанного пользователя."""
    memory_key = f"{thread_id}_{memory_type}"
    if memory_key in user_memories:
        user_memories[memory_key].clear()
        print(f"🗑️ История диалога для {thread_id} (тип: {memory_type}) очищена")

def get_conversation_history(thread_id: str = "default", memory_type: str = DEFAULT_MEMORY_TYPE):
    """Возвращает историю диалога для указанного пользователя."""
    memory_key = f"{thread_id}_{memory_type}"
    if memory_key in user_memories:
        return user_memories[memory_key].chat_memory.messages
    return []

def test_memory_types():
    """Тестирование различных типов памяти."""
    print("🧪 Тестирование типов памяти...")
    print("=" * 50)
    
    test_query = "Привет! Как дела?"
    memory_types = ["buffer", "window", "token_buffer", "summary"]
    
    for memory_type in memory_types:
        print(f"\n🔍 Тест памяти типа: {memory_type}")
        print("-" * 30)
        
        try:
            # Очищаем память перед тестом
            clear_conversation_history("test", memory_type)
            
            # Первый запрос
            result1 = call_agent(test_query, thread_id="test", memory_type=memory_type)
            print(f"✅ Первый ответ: {len(result1)} символов")
            
            # Второй запрос (должен учитывать память)
            result2 = call_agent("Что я спрашивал ранее?", thread_id="test", memory_type=memory_type)
            print(f"✅ Второй ответ: {len(result2)} символов")
            
            # Проверяем историю
            history = get_conversation_history("test", memory_type)
            print(f"📚 Сообщений в истории: {len(history)}")
            
        except Exception as e:
            print(f"❌ Ошибка с памятью {memory_type}: {e}")

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
            result = call_agent(query, thread_id="test_thread")
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
    test_memory = "--memory" in sys.argv
    
    if test_memory:
        test_memory_types()
        exit(0)
    
    if run_tests:
        test_simple_agent()
        exit(0)
    
    print("🚀 Упрощенный агент запущен!")
    print("Доступные типы памяти:")
    print("  - buffer: полная история (по умолчанию)")
    print("  - window: последние N сообщений")
    print("  - token_buffer: ограничение по токенам")
    print("  - summary: сжатая история")
    print("\nВведите 'exit', 'quit' или 'выход' для завершения")
    print("Или нажмите Ctrl+C для принудительной остановки")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("Спрашивай: ")
            
            # Проверка команд выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                print("Выход по команде пользователя")
                break
            
            result = call_agent(user_input, thread_id="main_thread")
            print(f"\n💬 Ответ:\n{result}")
            
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем (Ctrl+C)")
        print("До свидания!")
    except EOFError:
        print("\n\nПрограмма завершена (EOF)")
        print("До свидания!") 