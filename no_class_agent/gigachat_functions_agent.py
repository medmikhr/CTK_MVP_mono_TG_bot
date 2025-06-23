#!/usr/bin/env python3
"""
GigaChat Functions Agent (максимально упрощённая версия)
Рекомендуемый подход от Сбера для работы с инструментами
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage
from langgraph.prebuilt import create_react_agent
from pydantic import Field
from langchain_gigachat.tools.giga_tool import giga_tool
import time
import sys

from document_processor import search_documents

# Загрузка переменных окружения
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
logger.info("GigaChat LLM инициализирован")

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

@giga_tool(few_shot_examples=dama_few_shot_examples)
def dama_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в стандарте DAMA DMBOK")) -> str:
    return search_documents_tool(query, "dama_dmbok", "стандарте DAMA DMBOK")

@giga_tool(few_shot_examples=ctk_few_shot_examples)
def ctk_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в регламентах и методологических материалах ЦТК")) -> str:
    return search_documents_tool(query, "ctk_methodology", "регламентах и методологических материалах ЦТК")

def search_documents_tool(query: str, collection: str, collection_name: str) -> str:
    """Универсальная функция поиска документов."""
    try:
        logger.info(f"Поиск в {collection_name}: {query}")
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
        logger.error(f"Ошибка поиска в {collection_name}: {e}")
        return f"Ошибка при поиске в {collection_name}: {str(e)}"

functions = [dama_search, ctk_search]
logger.info("Функции для GigaChat настроены")

agent = create_react_agent(
    model=llm,
    tools=functions
)
logger.info("Агент с create_react_agent настроен")

def call_agent(query: str, user_id: str = "default") -> Dict[str, Any]:
    start_time = time.time()
    try:
        response = process_query(query, thread_id=user_id)
        end_time = time.time()
        processing_time = end_time - start_time
        return {
            "success": True,
            "response": response,
            "processing_time": processing_time,
            "thread_id": user_id
        }
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "response": "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже.",
            "error": error_msg,
            "processing_time": processing_time,
            "thread_id": user_id
        }

def process_query(user_query: str, thread_id: str = "default") -> str:
    try:
        logger.info(f"Обработка запроса: {user_query}")
        messages = [
            SystemMessage(content="""Ты - эксперт по управлению данными. У тебя есть доступ к двум источникам информации:
1. **Стандарт DAMA DMBOK** (Data Management Body Of Knowledge) - используй функцию dama_search для поиска информации о методологии управления данными, стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными согласно стандарту DAMA DMBOK.
2. **Регламенты и методологические материалы ЦТК** - используй функцию ctk_search для поиска информации о регламентах по процессам управления данными, политике данных для ДЗО (дочерних зависимых обществ), презентациях и других методологических документах по управлению данными от Центра технологического консалтинга (ЦТК).
**ВАЖНО**: Если пользователь спрашивает о методологии ЦТК, регламентах ЦТК, политиках данных для ДЗО, информационной архитектуре по методологии ЦТК - ОБЯЗАТЕЛЬНО используй функцию ctk_search.
Если пользователь спрашивает о стандарте DAMA DMBOK, методологии DAMA, областях управления данными по DAMA - ОБЯЗАТЕЛЬНО используй функцию dama_search.
Всегда используй соответствующие функции для поиска актуальной информации из документов. Дай подробный, структурированный ответ на русском языке."""),
            HumanMessage(content=user_query)
        ]
        used_tools = []
        tools_were_called = False
        while True:
            response = agent.invoke({"messages": messages}, config={"configurable": {"thread_id": thread_id}})
            if "tool_calls" in response and response["tool_calls"]:
                tools_were_called = True
                for tool_call in response["tool_calls"]:
                    func_name = tool_call["name"]
                    args = tool_call["args"]
                    logger.info(f"Выполняю функцию {func_name} с аргументами {args}")
                    if func_name == "dama_search":
                        used_tools.append("Стандарт DAMA DMBOK")
                        result = dama_search.invoke(args)
                    elif func_name == "ctk_search":
                        used_tools.append("Регламенты и материалы ЦТК")
                        result = ctk_search.invoke(args)
                    else:
                        result = None
                    messages.append(FunctionMessage(name=func_name, content=result))
            else:
                bot_answer = response["messages"][-1].content
                logger.info("Запрос обработан успешно")
                if tools_were_called and used_tools:
                    unique_tools = list(set(used_tools))
                    tools_info = f"\n\n🔍 **Источники информации:** {', '.join(unique_tools)}"
                    return bot_answer + tools_info
                else:
                    tools_info = "\n\n💡 **Ответ основан на общих знаниях** (без использования документов)"
                    return bot_answer + tools_info
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        try:
            logger.info("Используем fallback - запрос к LLM с функциями")
            llm_with_functions = llm.bind_tools(functions)
            response = llm_with_functions.invoke(messages)
            return response.content + "\n\n⚠️ **Использован fallback режим** (возможны ошибки в работе инструментов)"
        except Exception as fallback_error:
            logger.error(f"Ошибка fallback: {fallback_error}")
            try:
                response = llm.invoke(messages)
                return response.content + "\n\n⚠️ **Использован аварийный режим** (инструменты недоступны)"
            except Exception as final_error:
                logger.error(f"Финальная ошибка fallback: {final_error}")
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
        print("🚀 GigaChat Functions Agent готов к работе!")
        print("Введите 'exit', 'quit' или 'выход' для завершения")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nСпрашивай: ")
                if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                    print("Выход по команде пользователя")
                    break
                if not user_input.strip():
                    continue
                print(f"User: {user_input}")
                start_time = time.time()
                bot_answer = process_query(user_input, thread_id="main_thread")
                end_time = time.time()
                print(f"\n💬 Bot (за {end_time - start_time:.2f}с):")
                print(f"\033[93m{bot_answer}\033[0m")
            except KeyboardInterrupt:
                print("\n\nПрограмма завершена пользователем (Ctrl+C)")
                break
            except EOFError:
                print("\n\nПрограмма завершена (EOF)")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
    except Exception as e:
        print(f"❌ Критическая ошибка инициализации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 