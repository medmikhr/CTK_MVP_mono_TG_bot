#!/usr/bin/env python3
"""
GigaChat Functions Agent (максимально упрощённая версия)
Рекомендуемый подход от Сбера для работы с инструментами
"""

import os
import time
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage
from langgraph.prebuilt import create_react_agent
from pydantic import Field
from langchain_gigachat.tools.giga_tool import giga_tool

from document_processor import search_documents

# Загрузка переменных окружения
load_dotenv()

# Инициализация компонентов один раз при запуске
gc_auth = os.getenv('GIGACHAT_TOKEN')
if not gc_auth:
    raise ValueError("Не найден токен GigaChat в переменных окружения")

llm = GigaChat(
    credentials=gc_auth,
    model='GigaChat:latest',
    verify_ssl_certs=False,
    profanity_check=False
)

# Создание функций с декоратором giga_tool
dama_few_shot_examples = [
    {"request": "Найди информацию о методологии управления данными в стандарте DAMA DMBOK", "params": {"query": "методология управления данными стандарт DAMA DMBOK", "collection": "dama_dmbok"}},
    {"request": "Что такое DAMA DMBOK?", "params": {"query": "DAMA DMBOK Data Management Body Of Knowledge стандарт", "collection": "dama_dmbok"}},
    {"request": "Расскажи о ролях в управлении данными согласно стандарту DAMA DMBOK", "params": {"query": "роли ответственность управление данными стандарт DAMA DMBOK", "collection": "dama_dmbok"}}
]

ctk_few_shot_examples = [
    {"request": "Найди регламенты по процессам управления данными", "params": {"query": "регламенты процессы управления данными", "collection": "ctk_methodology"}},
    {"request": "Расскажи о политике данных для ДЗО", "params": {"query": "политика данных дочерние зависимые общества ДЗО", "collection": "ctk_methodology"}},
    {"request": "Что такое ЦТК и какие документы они предоставляют?", "params": {"query": "ЦТК центральный технологический консалтинг методологические документы", "collection": "ctk_methodology"}}
]

sbf_few_shot_examples = [
    {"request": "Найди информацию о метаданных в СберФакторинг", "params": {"query": "метаданные структура данных СберФакторинг СБФ", "collection": "sbf_meta"}},
    {"request": "Расскажи о структуре данных компании СберФакторинг", "params": {"query": "структура данных модели СберФакторинг СБФ", "collection": "sbf_meta"}},
    {"request": "Какие метаданные используются в СБФ?", "params": {"query": "метаданные СберФакторинг СБФ синтезированные данные", "collection": "sbf_meta"}}
]

@giga_tool(few_shot_examples=dama_few_shot_examples)
def dama_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в стандарте DAMA DMBOK")) -> str:
    """ВСЕГДА используй эту функцию, если в запросе упоминаются: DAMA, DMBOK, dmbok, Data Management Body Of Knowledge, области управления данными, роли управления данными, стандарт DAMA. НЕ ОТВЕЧАЙ на основе общих знаний - ВСЕГДА ищи в документах."""
    return search_documents_tool(query, "dama_dmbok", "стандарте DAMA DMBOK")

@giga_tool(few_shot_examples=ctk_few_shot_examples)
def ctk_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в регламентах и методологических материалах ЦТК")) -> str:
    """ВСЕГДА используй эту функцию, если в запросе упоминаются: ЦТК, центральный технологический консалтинг, методология ЦТК, слои информационной архитектуры, регламенты ЦТК, политика данных ДЗО. НЕ ОТВЕЧАЙ на основе общих знаний - ВСЕГДА ищи в документах."""
    return search_documents_tool(query, "ctk_methodology", "регламентах и методологических материалах ЦТК")

@giga_tool(few_shot_examples=sbf_few_shot_examples)
def sbf_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в синтезированных метаданных компании СберФакторинг (СБФ)")) -> str:
    """ВСЕГДА используй эту функцию, если в запросе упоминаются: СберФакторинг, СБФ, метаданные СБФ, структура данных СберФакторинг. НЕ ОТВЕЧАЙ на основе общих знаний - ВСЕГДА ищи в документах."""
    return search_documents_tool(query, "sbf_meta", "синтезированных метаданных компании СберФакторинг (СБФ)")

def search_documents_tool(query: str, collection: str, collection_name: str) -> str:
    """Универсальная функция поиска документов."""
    try:
        results = search_documents(query, collection, n_results=5)
        if not results:
            return f"Информация по данному запросу не найдена в {collection_name}."
        content_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Неизвестный источник')
            score = result['score']
            content_parts.append(f"Источник {i}: {source} (релевантность: {score:.3f})\n{result['text']}")
        return "\n\n---\n\n".join(content_parts)
    except Exception as e:
        return f"Ошибка при поиске в {collection_name}: {str(e)}"

functions = [dama_search, ctk_search, sbf_search]

agent = create_react_agent(
    model=llm,
    tools=functions
)

def call_agent(query: str, user_id: str = "default") -> str:
    try:
        config = {"configurable": {"thread_id": user_id}}
        final_response = ""
        
        # Добавляем системный промпт с инструкциями
        system_message = """Ты - эксперт по управлению данными. У тебя есть доступ к трём функциям поиска:

1. dama_search - для поиска в стандарте DAMA DMBOK
2. ctk_search - для поиска в регламентах и материалах ЦТК  
3. sbf_search - для поиска в метаданных СберФакторинг

КРИТИЧЕСКИ ВАЖНО: Если в запросе пользователя есть любые из следующих ключевых слов, ты ОБЯЗАТЕЛЬНО должен использовать соответствующую функцию:

Для dama_search: DAMA, DMBOK, dmbok, "области управления данными", "роли управления данными", "стандарт DAMA", "Data Management Body of Knowledge"

Для ctk_search: ЦТК, "центральный технологический консалтинг", "методология ЦТК", "слои информационной архитектуры", "регламенты ЦТК", "политика данных ДЗО"

Для sbf_search: СберФакторинг, СБФ, "метаданные СБФ", "структура данных СберФакторинг"

НЕ ОТВЕЧАЙ на основе общих знаний, если в запросе есть эти ключевые слова. ВСЕГДА используй функции для поиска актуальной информации из документов.

Если ключевых слов нет, можешь отвечать на основе своих знаний."""
        
        for event in agent.stream(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
            },
            stream_mode="values",
            config=config,
        ):
            if "messages" in event and event["messages"]:
                for message in event["messages"]:
                    if hasattr(message, '__class__'):
                        message_type = message.__class__.__name__
                        print(f"\n=== {message_type} ===")
                        if hasattr(message, 'content') and message.content:
                            print(f"Content: {message.content}")
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            print(f"Tool calls: {message.tool_calls}")
                        if hasattr(message, 'tool_call_id') and message.tool_call_id:
                            print(f"Tool call ID: {message.tool_call_id}")
                        if hasattr(message, 'name') and message.name:
                            print(f"Name: {message.name}")
                        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                            print(f"Additional kwargs: {message.additional_kwargs}")
            
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content'):
                    final_response = last_message.content
        
        return final_response
    except Exception as e:
        return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"

def get_functions_info() -> Dict[str, Any]:
    """Получение информации о доступных функциях для бота."""
    return {
        "total_functions": len(functions),
        "function_names": [func.name for func in functions],
        "functions": [
            {
                "name": func.name,
                "description": func.description,
                "args_schema": func.args_schema.schema() if hasattr(func, 'args_schema') else None
            }
            for func in functions
        ]
    }

def main():
    try:
        while True:
            try:
                user_input = input("Спрашивай: ")
                if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                    break
                if not user_input.strip():
                    continue
                print(f"User: {user_input}")
                start_time = time.time()
                bot_answer = call_agent(user_input, user_id="main_thread")
                end_time = time.time()
                print(f"\nBot (за {end_time - start_time:.2f}с):")
                print(f"{bot_answer}")
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Критическая ошибка инициализации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 