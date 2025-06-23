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
print("✅ GigaChat LLM инициализирован")

# Конфигурация коллекций
COLLECTIONS_CONFIG = {
    "dama_dmbok": {
        "name": "стандарте DAMA DMBOK",
        "function_name": "dama_search",
        "keywords": "DAMA, DMBOK, dmbok, Data Management Body Of Knowledge, области управления данными, роли управления данными, стандарт DAMA",
        "description": "Поиск информации в стандарте DAMA DMBOK по заданному запросу. Используй эту функцию, если в запросе упоминаются: DAMA, DMBOK, dmbok, Data Management Body Of Knowledge, области управления данными, роли управления данными, стандарт DAMA."
    },
    "ctk_methodology": {
        "name": "регламентах и методологических материалах ЦТК",
        "function_name": "ctk_search", 
        "keywords": "ЦТК, центральный технологический консалтинг, методология ЦТК, слои информационной архитектуры, регламенты ЦТК, политика данных ДЗО",
        "description": "Поиск информации в регламентах и методологических материалах ЦТК по заданному запросу. Используй эту функцию, если в запросе упоминаются: ЦТК, центральный технологический консалтинг, методология ЦТК, слои информационной архитектуры, регламенты ЦТК, политика данных ДЗО."
    },
    "sbf_meta": {
        "name": "синтезированных метаданных компании СберФакторинг (СБФ)",
        "function_name": "sbf_search",
        "keywords": "СберФакторинг, СБФ, метаданные СБФ, структура данных СберФакторинг",
        "description": "Поиск информации в синтезированных метаданных компании СберФакторинг (СБФ) по заданному запросу. Используй эту функцию, если в запросе упоминаются: СберФакторинг, СБФ, метаданные СБФ, структура данных СберФакторинг."
    }
}

# Примеры для DAMA поиска
dama_dmbok_few_shot_examples = [
    {
        "request": "Найди информацию о методологии управления данными в стандарте DAMA DMBOK",
        "params": {"query": "методология управления данными стандарт DAMA DMBOK"}
    },
    {
        "request": "Что такое DAMA DMBOK?",
        "params": {"query": "DAMA DMBOK Data Management Body Of Knowledge стандарт"}
    },
    {
        "request": "Расскажи о ролях в управлении данными согласно стандарту DAMA DMBOK",
        "params": {"query": "роли ответственность управление данными стандарт DAMA DMBOK"}
    }
]

# Примеры для ЦТК поиска
ctk_methodology_few_shot_examples = [
    {
        "request": "Найди регламенты по процессам управления данными",
        "params": {"query": "регламенты процессы управления данными"}
    },
    {
        "request": "Расскажи о политике данных для ДЗО",
        "params": {"query": "политика данных дочерние зависимые общества ДЗО"}
    },
    {
        "request": "Что такое ЦТК и какие документы они предоставляют?",
        "params": {"query": "ЦТК центральный технологический консалтинг методологические документы"}
    }
]

# Примеры для СБФ поиска
sbf_meta_few_shot_examples = [
    {
        "request": "Какие метаданные используются в СберФакторинг?",
        "params": {"query": "метаданные структура данных СберФакторинг СБФ"}
    },
    {
        "request": "Расскажи о структуре данных в СБФ",
        "params": {"query": "структура данных метаданные СберФакторинг"}
    },
    {
        "request": "Какие системы используются в СберФакторинг?",
        "params": {"query": "системы метаданные СБФ СберФакторинг"}
    }
]

def create_search_function(collection: str, collection_name: str, function_name: str):
    """Создает универсальную функцию поиска с логированием."""
    
    @giga_tool(few_shot_examples=globals()[f"{collection}_few_shot_examples"])
    def search_func(query: str = Field(description=f"Поисковый запрос на русском языке для поиска в {collection_name}")) -> str:
        """Универсальная функция поиска с автоматическим логированием."""
        print(f"\n🔍 ВЫЗОВ ИНСТРУМЕНТА: {function_name}")
        print(f"📝 Запрос: {query}")
        result = search_documents_tool(query, collection, collection_name)
        print(f"📄 Результат {function_name} ({len(result)} символов):")
        print(f"---\n{result}\n---")
        return result
    
    # Устанавливаем правильное имя функции
    search_func.__name__ = function_name
    return search_func

# Создаем функции поиска с помощью универсальной функции
functions = []
for collection, config in COLLECTIONS_CONFIG.items():
    func = create_search_function(collection, config["name"], config["function_name"])
    func.__doc__ = config["description"]
    globals()[config["function_name"]] = func
    functions.append(func)

def search_documents_tool(query: str, collection: str, collection_name: str) -> str:
    """Универсальная функция поиска документов."""
    try:
        print(f"🔎 Поиск в {collection_name}: {query}")
        results = search_documents(query, collection, n_results=5)
        print(f"📊 Получено результатов: {len(results) if results else 0}")
        
        if not results:
            print(f"⚠️ Результаты не найдены в {collection_name}")
            return f"Информация по данному запросу не найдена в {collection_name}."
        
        content_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Неизвестный источник')
            score = result['score']
            content_parts.append(f"Источник {i}: {source} (релевантность: {score:.3f})\n{result['text']}")
            print(f"📋 Результат {i}: {source} (score: {score:.3f})")
        
        final_result = "\n\n---\n\n".join(content_parts)
        print(f"✅ Финальный результат для {collection_name}: {len(final_result)} символов")
        return final_result
    except Exception as e:
        print(f"❌ Ошибка поиска в {collection_name}: {e}")
        return f"Ошибка при поиске в {collection_name}: {str(e)}"

print("✅ Функции для GigaChat настроены")

# Автоматическая генерация системного промпта
def generate_system_prompt():
    """Генерирует системный промпт на основе конфигурации коллекций."""
    prompt_parts = [
        "Ты - эксперт по управлению данными. У тебя есть доступ к источникам информации через специальные функции.\n\n**ПРАВИЛА ИСПОЛЬЗОВАНИЯ ФУНКЦИЙ:**\n"
    ]
    
    for i, (collection, config) in enumerate(COLLECTIONS_CONFIG.items(), 1):
        prompt_parts.append(
            f"{i}. **{config['function_name']}** - для запросов о {config['name']}\n"
            f"   - Ключевые слова: {config['keywords']}\n"
            f"   - Примеры запросов: см. few-shot примеры функции\n"
        )
    
    prompt_parts.extend([
        "\n**ВАЖНО:**",
        "- Если в запросе есть любое из ключевых слов - ОБЯЗАТЕЛЬНО используй соответствующую функцию",
        "- У каждой функции есть few-shot примеры, которые показывают, как правильно формулировать запросы",
        "- НЕ ОТВЕЧАЙ на основе общих знаний, если есть возможность использовать функции",
        "- Всегда используй функции для поиска актуальной информации из документов",
        "\n**ПРИМЕРЫ РАБОТЫ:**",
        "- Запрос \"Что такое DAMA DMBOK?\" → используй dama_search с запросом \"DAMA DMBOK Data Management Body Of Knowledge стандарт\"",
        "- Запрос \"Расскажи о политике данных для ДЗО\" → используй ctk_search с запросом \"политика данных дочерние зависимые общества ДЗО\"",
        "- Запрос \"Какие метаданные в СБФ?\" → используй sbf_search с запросом \"метаданные структура данных СберФакторинг СБФ\"",
        "\nДай подробный, структурированный ответ на русском языке."
    ])
    
    return "\n".join(prompt_parts)

# Создание промпта для агента
prompt = ChatPromptTemplate.from_messages([
    ("system", generate_system_prompt()),
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
print("✅ Агент с create_tool_calling_agent настроен")

def call_agent(user_query: str, thread_id: str = "default") -> str:
    try:
        print(f"\n🤖 ОБРАБОТКА ЗАПРОСА: {user_query}")
        
        # Используем агента для обработки запроса
        print("🎯 Используем create_tool_calling_agent...")
        result = agent_executor.invoke({
            "input": user_query,
            "chat_history": []
        })
        
        response = result.get("output", "Не удалось получить ответ")
        print(f"💬 Ответ агента ({len(response)} символов):")
        print(f"---\n{response}\n---")
        
        # Проверяем, были ли вызваны инструменты
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print("🔍 Инструменты были вызваны через агента")
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
        print("💡 Инструменты не были вызваны, ответ основан на общих знаниях")
        tools_info = "\n\n💡 **Ответ основан на общих знаниях** (без использования документов)"
        return response + tools_info
        
    except Exception as e:
        print(f"❌ Ошибка обработки запроса: {e}")
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
        print("🚀 GigaChat Tool Calling Agent готов к работе!")
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
                print(f"👤 User: {user_input}")
                start_time = time.time()
                bot_answer = call_agent(user_input, thread_id="main_thread")
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