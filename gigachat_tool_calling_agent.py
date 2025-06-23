#!/usr/bin/env python3
"""
GigaChat Tool Calling Agent (максимально упрощённая версия)
Рекомендуемый подход от Сбера для работы с инструментами
"""

import os
import time
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
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

# Few-shot examples для функций
few_shot_dama = [
    {
        "request": "Что такое DMBOK?",
        "params": {"query": "Что такое DMBOK?"}
    },
    {
        "request": "Какие области управления данными согласно стандарту DAMA?",
        "params": {"query": "Какие области управления данными согласно стандарту DAMA?"}
    }
]

few_shot_ctk = [
    {
        "request": "Какие есть слои информационной архитектуры согласно методологии ЦТК?",
        "params": {"query": "Какие есть слои информационной архитектуры согласно методологии ЦТК?"}
    },
    {
        "request": "Расскажи о методологии ЦТК",
        "params": {"query": "Расскажи о методологии ЦТК"}
    }
]

few_shot_sbf = [
    {
        "request": "Какие метаданные используются в СберФакторинг?",
        "params": {"query": "Какие метаданные используются в СберФакторинг?"}
    },
    {
        "request": "Расскажи о структуре данных СберФакторинг",
        "params": {"query": "Расскажи о структуре данных СберФакторинг"}
    }
]

# Создание функций с декоратором giga_tool
@giga_tool(few_shot_examples=few_shot_dama)
def dama_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в стандарте DAMA DMBOK")) -> str:
    """Поиск информации в стандарте DAMA DMBOK по заданному запросу. Используй эту функцию, если в запросе упоминаются: DAMA, DMBOK, dmbok, Data Management Body Of Knowledge, области управления данными, роли управления данными, стандарт DAMA."""
    result = search_documents_tool(query, "dama_dmbok", "стандарте DAMA DMBOK")
    return result

@giga_tool(few_shot_examples=few_shot_ctk)
def ctk_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в регламентах и методологических материалах ЦТК")) -> str:
    """Поиск информации в регламентах и методологических материалах ЦТК по заданному запросу. Используй эту функцию, если в запросе упоминаются: ЦТК, Центр Технологического Консалтинга, методология ЦТК, слои информационной архитектуры, регламенты ЦТК, политика данных ДЗО."""
    result = search_documents_tool(query, "ctk_methodology", "регламентах и методологических материалах ЦТК")
    return result

@giga_tool(few_shot_examples=few_shot_sbf)
def sbf_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в синтезированных метаданных компании СберФакторинг (СБФ)")) -> str:
    """Поиск информации в синтезированных метаданных компании СберФакторинг (СБФ) по заданному запросу. Используй эту функцию, если в запросе упоминаются: СберФакторинг, СБФ, метаданные СБФ, структура данных СберФакторинг."""
    result = search_documents_tool(query, "sbf_meta", "синтезированных метаданных компании СберФакторинг (СБФ)")
    return result

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
            # Убираем лишние переносы строк из текста
            clean_text = result['text'].replace('\n', ' ').replace('  ', ' ').strip()
            content_parts.append(f"\nИсточник {i}: {source} (релевантность: {score:.3f})\n{clean_text}")
        
        final_result = "\n\n---\n".join(content_parts)
        return final_result
    except Exception as e:
        return f"Ошибка при поиске в {collection_name}: {str(e)}"

functions = [dama_search, ctk_search, sbf_search]

# Создание промпта для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты - эксперт по управлению данными. У тебя есть доступ к трём источникам информации через специальные функции.

**ПРАВИЛА ИСПОЛЬЗОВАНИЯ ФУНКЦИЙ:**

1. **dama_search** - для запросов о DAMA DMBOK
   - Ключевые слова: DAMA, DMBOK, dmbok, Data Management Body Of Knowledge, области управления данными, роли управления данными, стандарт DAMA
   - Примеры: "Что такое DMBOK?", "Какие области управления данными согласно стандарту dmbok"

2. **ctk_search** - для запросов о ЦТК
   - Ключевые слова: ЦТК, Центр Технологического Консалтинга, методология ЦТК, слои информационной архитектуры, регламенты ЦТК, политика данных ДЗО
   - Примеры: "Какие есть слои информационной архитектуры согласно методологии ЦТК", "Расскажи о методологии ЦТК"

3. **sbf_search** - для запросов о СберФакторинг
   - Ключевые слова: СберФакторинг, СБФ, метаданные СБФ, структура данных СберФакторинг
   - Примеры: "Какие метаданные используются в СберФакторинг?"

**ВАЖНО:** Если в запросе есть любое из ключевых слов - ОБЯЗАТЕЛЬНО используй соответствующую функцию. НЕ ОТВЕЧАЙ на основе общих знаний.

Всегда используй функции для поиска актуальной информации из документов."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Создание агента с create_tool_calling_agent
agent = create_tool_calling_agent(llm, functions, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=functions,
    return_intermediate_steps=True,
    verbose=True
)

def call_agent(user_query: str, thread_id: str = "default") -> str:
    try:
        result = agent_executor.invoke({
            "input": user_query,
            "chat_history": []
        })
        
        response = result.get("output", "Не удалось получить ответ")
        
        # Проверяем, были ли вызваны инструменты
        if "intermediate_steps" in result and result["intermediate_steps"]:
            # Определяем какие инструменты были использованы
            used_tools = []
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    tool_name = step[0].tool
                    used_tools.append(tool_name)
            
            if used_tools:
                unique_tools = list(set(used_tools))
                tools_info = f"\n\n🔍 **Источники информации:** {', '.join(unique_tools)}"
                return response + tools_info
        
        # Если инструменты не были вызваны
        tools_info = "\n\n💡 **Ответ основан на общих знаниях** (без использования документов)"
        return response + tools_info
        
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
                user_input = input("\nСпрашивай: ")
                if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                    break
                if not user_input.strip():
                    continue
                print(f"👤 User: {user_input}")
                start_time = time.time()
                bot_answer = call_agent(user_input, thread_id="main_thread")
                end_time = time.time()
                print(f"\n💬 Bot (за {end_time - start_time:.2f}с):")
                print(f"\033[93m{bot_answer}\033[0m")
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
    except Exception as e:
        print(f"❌ Критическая ошибка инициализации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
    