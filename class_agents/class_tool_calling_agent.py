#!/usr/bin/env python3
"""
GigaChat Tool Calling Agent - современный агент с использованием create_tool_calling_agent
Рекомендуемый подход для работы с function calling от LangChain
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_gigachat.tools.giga_tool import giga_tool
import time
import sys
import json

# Импортируем общие компоненты
from document_processor_langchain import get_vectorstore, get_document_info

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Модели для результатов функций
class DamaSearchResult(BaseModel):
    """Результат поиска в документах DAMA DMBOK."""
    content: str = Field(description="Найденная информация из документов DAMA DMBOK")
    sources: List[str] = Field(description="Список источников информации")

class CtkSearchResult(BaseModel):
    """Результат поиска в методологических материалах ЦТК."""
    content: str = Field(description="Найденная информация из материалов ЦТК")
    sources: List[str] = Field(description="Список источников информации")

class AgentResponse(BaseModel):
    """Ответ агента для использования в боте."""
    success: bool = Field(description="Успешность выполнения запроса")
    response: str = Field(description="Ответ агента")
    error: Optional[str] = Field(description="Описание ошибки, если есть", default=None)
    processing_time: float = Field(description="Время обработки запроса в секундах")
    thread_id: str = Field(description="ID потока для памяти")

# Глобальная переменная для хранения экземпляра агента
_agent_instance = None

def get_agent_instance() -> 'GigaChatToolCallingAgent':
    """Получение или создание экземпляра агента (синглтон)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GigaChatToolCallingAgent()
    return _agent_instance

def call_agent(query: str, user_id: str = "default") -> AgentResponse:
    """
    Удобная функция для вызова агента из бота.
    
    Args:
        query: Запрос пользователя
        user_id: ID пользователя для создания уникального потока памяти
        
    Returns:
        AgentResponse с результатом выполнения
    """
    start_time = time.time()
    
    try:
        # Получаем экземпляр агента
        agent = get_agent_instance()
        
        # Обрабатываем запрос
        response = agent.process_query(query, thread_id=user_id)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return AgentResponse(
            success=True,
            response=response,
            processing_time=processing_time,
            thread_id=user_id
        )
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg)
        
        return AgentResponse(
            success=False,
            response="Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже.",
            error=error_msg,
            processing_time=processing_time,
            thread_id=user_id
        )

def get_agent_status() -> Dict[str, Any]:
    """
    Получение статуса агента для мониторинга.
    
    Returns:
        Словарь с информацией о состоянии агента
    """
    try:
        agent = get_agent_instance()
        store_info = agent.get_store_info()
        functions_info = agent.get_functions_info()
        
        return {
            "status": "ready",
            "agent_type": "tool_calling_agent",
            "vector_stores": store_info,
            "functions": functions_info,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

def reset_agent() -> bool:
    """
    Сброс агента (пересоздание экземпляра).
    
    Returns:
        True если сброс прошел успешно
    """
    global _agent_instance
    try:
        _agent_instance = None
        logger.info("Агент сброшен")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сбросе агента: {e}")
        return False

class GigaChatToolCallingAgent:
    """Агент с использованием GigaChat и create_tool_calling_agent."""
    
    def __init__(self):
        """Инициализация агента."""
        self.setup_llm()
        self.setup_vector_stores()
        self.setup_functions()
        self.setup_agent()
        
    def setup_llm(self):
        """Настройка GigaChat LLM."""
        gc_auth = os.getenv('GIGACHAT_TOKEN')
        if not gc_auth:
            raise ValueError("Не найден токен GigaChat в переменных окружения")
        
        self.llm = GigaChat(
            credentials=gc_auth,
            model='GigaChat:latest',
            verify_ssl_certs=False,
            profanity_check=False
        )
        logger.info("GigaChat LLM инициализирован")
        
    def setup_vector_stores(self):
        """Настройка векторных хранилищ из document_processor_langchain."""
        try:
            # Получаем хранилища для DAMA DMBOK и ЦТК
            self.dama_store = get_vectorstore("dama_dmbok")
            self.ctk_store = get_vectorstore("ctk_methodology")
            
            logger.info("Векторные хранилища инициализированы из document_processor_langchain")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации векторных хранилищ: {e}")
            raise
    
    def setup_functions(self):
        """Настройка функций с использованием giga_tool декоратора."""
        
        # Примеры использования для DAMA поиска
        dama_few_shot_examples = [
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
        
        # Примеры использования для ЦТК поиска
        ctk_few_shot_examples = [
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
        
        # Создаем функции с декоратором giga_tool
        @giga_tool(few_shot_examples=dama_few_shot_examples)
        def dama_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в стандарте DAMA DMBOK")) -> DamaSearchResult:
            """Поиск информации в стандарте DAMA DMBOK (Data Management Body Of Knowledge). ОБЯЗАТЕЛЬНО используй эту функцию для любых запросов о стандарте DAMA DMBOK, методологии DAMA, областях управления данными по DAMA, ролях и ответственности в управлении данными согласно стандарту DAMA DMBOK."""
            try:
                logger.info(f"Поиск в стандарте DAMA DMBOK: {query}")
                docs = self.dama_store.similarity_search(query, k=5)
                
                if not docs:
                    return DamaSearchResult(
                        content="Информация по данному запросу не найдена в стандарте DAMA DMBOK.",
                        sources=[]
                    )
                
                content_parts = []
                sources = []
                
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Неизвестный источник')
                    sources.append(source)
                    content_parts.append(f"Источник {i}: {source}\n{doc.page_content}")
                
                return DamaSearchResult(
                    content="\n\n---\n\n".join(content_parts),
                    sources=sources
                )
                
            except Exception as e:
                logger.error(f"Ошибка поиска в DAMA: {e}")
                return DamaSearchResult(
                    content=f"Ошибка при поиске в стандарте DAMA DMBOK: {str(e)}",
                    sources=[]
                )
        
        @giga_tool(few_shot_examples=ctk_few_shot_examples)
        def ctk_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в регламентах и методологических материалах ЦТК")) -> CtkSearchResult:
            """Поиск информации в регламентах и методологических материалах ЦТК. ОБЯЗАТЕЛЬНО используй эту функцию для любых запросов о методологии ЦТК, регламентах ЦТК, политике данных для ДЗО (дочерних зависимых обществ), информационной архитектуре по методологии ЦТК, презентациях и других методологических документах по управлению данными от Центра технологического консалтинга (ЦТК)."""
            try:
                logger.info(f"Поиск в ЦТК: {query}")
                docs = self.ctk_store.similarity_search(query, k=5)
                
                if not docs:
                    return CtkSearchResult(
                        content="Информация по данному запросу не найдена в регламентах и методологических материалах ЦТК.",
                        sources=[]
                    )
                
                content_parts = []
                sources = []
                
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Неизвестный источник')
                    sources.append(source)
                    content_parts.append(f"Источник {i}: {source}\n{doc.page_content}")
                
                return CtkSearchResult(
                    content="\n\n---\n\n".join(content_parts),
                    sources=sources
                )
                
            except Exception as e:
                logger.error(f"Ошибка поиска в ЦТК: {e}")
                return CtkSearchResult(
                    content=f"Ошибка при поиске в материалах ЦТК: {str(e)}",
                    sources=[]
                )
        
        # Сохраняем функции
        self.dama_search_func = dama_search
        self.ctk_search_func = ctk_search
        self.functions = [dama_search, ctk_search]
        
        logger.info("Функции для GigaChat настроены")
    
    def setup_agent(self):
        """Настройка агента с использованием create_tool_calling_agent."""
        try:
            # Создаем промпт для агента
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Ты - эксперт по управлению данными. У тебя есть доступ к двум источникам информации:

1. **Стандарт DAMA DMBOK** (Data Management Body Of Knowledge) - используй функцию dama_search для поиска информации о методологии управления данными, стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными согласно стандарту DAMA DMBOK.

2. **Регламенты и методологические материалы ЦТК** - используй функцию ctk_search для поиска информации о регламентах по процессам управления данными, политике данных для ДЗО (дочерних зависимых обществ), презентациях и других методологических документах по управлению данными от Центра технологического консалтинга (ЦТК).

**ВАЖНО**: Если пользователь спрашивает о методологии ЦТК, регламентах ЦТК, политиках данных для ДЗО, информационной архитектуре по методологии ЦТК - ОБЯЗАТЕЛЬНО используй функцию ctk_search.

Если пользователь спрашивает о стандарте DAMA DMBOK, методологии DAMA, областях управления данными по DAMA - ОБЯЗАТЕЛЬНО используй функцию dama_search.

Всегда используй соответствующие функции для поиска актуальной информации из документов. Дай подробный, структурированный ответ на русском языке."""),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # Создаем агент с create_tool_calling_agent
            self.agent = create_tool_calling_agent(self.llm, self.functions, prompt)
            self.agent_executor = AgentExecutor(agent=self.agent, tools=self.functions)
            self.memory = MemorySaver()
            
            logger.info("Агент с create_tool_calling_agent настроен")
        except Exception as e:
            logger.error(f"Ошибка настройки агента: {e}")
            raise
    
    def process_query(self, user_query: str, thread_id: str = "default") -> str:
        """
        Обработка запроса пользователя с использованием агента.
        """
        try:
            logger.info(f"Обработка запроса: {user_query}")
            
            # Отслеживаем использованные инструменты
            used_tools = []
            tools_were_called = False
            
            # Выполняем запрос через AgentExecutor
            result = self.agent_executor.invoke({
                "input": user_query,
                "chat_history": []
            }, config={"configurable": {"thread_id": thread_id}})
            
            # Получаем ответ
            response_text = result.get("output", "Не удалось получить ответ")
            
            # Проверяем, были ли вызваны инструменты
            # В create_tool_calling_agent это можно отследить через intermediate_steps
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        action = step[0]
                        if hasattr(action, 'tool') and action.tool:
                            tools_were_called = True
                            if action.tool == "dama_search":
                                used_tools.append("Стандарт DAMA DMBOK")
                            elif action.tool == "ctk_search":
                                used_tools.append("Регламенты и материалы ЦТК")
            
            logger.info("Запрос обработан успешно")
            
            # Добавляем информацию об использованных инструментах
            if tools_were_called and used_tools:
                unique_tools = list(set(used_tools))  # Убираем дубликаты
                tools_info = f"\n\n🔍 **Источники информации:** {', '.join(unique_tools)}"
                return response_text + tools_info
            else:
                tools_info = "\n\n💡 **Ответ основан на общих знаниях** (без использования документов)"
                return response_text + tools_info
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            
            # Fallback к простому запросу с функциями
            try:
                logger.info("Используем fallback - запрос к LLM с функциями")
                llm_with_functions = self.llm.bind_tools(self.functions)
                messages = [
                    SystemMessage(content="Ты эксперт по управлению данными. Используй доступные функции для поиска информации."),
                    HumanMessage(content=user_query)
                ]
                response = llm_with_functions.invoke(messages)
                return response.content + "\n\n⚠️ **Использован fallback режим** (возможны ошибки в работе инструментов)"
            except Exception as fallback_error:
                logger.error(f"Ошибка fallback: {fallback_error}")
                # Последний fallback - простой LLM без функций
                try:
                    messages = [
                        SystemMessage(content="Ты эксперт по управлению данными."),
                        HumanMessage(content=user_query)
                    ]
                    response = self.llm.invoke(messages)
                    return response.content + "\n\n⚠️ **Использован аварийный режим** (инструменты недоступны)"
                except Exception as final_error:
                    logger.error(f"Финальная ошибка fallback: {final_error}")
                    return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
    
    def get_store_info(self) -> Dict[str, Any]:
        """Получение информации о векторных хранилищах через document_processor_langchain."""
        try:
            # Получаем информацию о коллекциях через document_processor_langchain
            dama_info = get_document_info("dama_dmbok")
            ctk_info = get_document_info("ctk_methodology")
            
            return {
                "dama_dmbok": {
                    "documents": dama_info.get("total_documents", 0),
                    "status": "ready" if dama_info.get("total_documents", 0) > 0 else "empty",
                    "details": dama_info
                },
                "ctk_methodology": {
                    "documents": ctk_info.get("total_documents", 0),
                    "status": "ready" if ctk_info.get("total_documents", 0) > 0 else "empty",
                    "details": ctk_info
                }
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о хранилищах: {e}")
            return {"error": str(e)}
    
    def get_functions_info(self) -> Dict[str, Any]:
        """Получение информации о доступных функциях."""
        return {
            "total_functions": len(self.functions),
            "function_names": [func.name for func in self.functions],
            "functions": [
                {
                    "name": func.name,
                    "description": func.description,
                    "args_schema": func.args_schema.schema() if hasattr(func, 'args_schema') else None
                }
                for func in self.functions
            ]
        }

def main():
    """Главная функция для запуска GigaChat Tool Calling Agent."""
    try:
        print("🚀 Инициализация GigaChat Tool Calling Agent...")
        agent = GigaChatToolCallingAgent()
        
        print("\n✅ GigaChat Tool Calling Agent готов к работе!")
        print("Введите 'exit', 'quit' или 'выход' для завершения")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nСпрашивай: ")
                
                # Проверка команд выхода
                if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                    print("Выход по команде пользователя")
                    break
                
                if not user_input.strip():
                    continue
                
                print(f"User: {user_input}")
                
                # Используем process_query для обработки запроса
                start_time = time.time()
                bot_answer = agent.process_query(user_input, thread_id="main_thread")
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