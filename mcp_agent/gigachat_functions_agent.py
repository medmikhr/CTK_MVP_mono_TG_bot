#!/usr/bin/env python3
"""
GigaChat Functions Agent - агент с использованием function calling
Рекомендуемый подход от Сбера для работы с инструментами
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_gigachat.tools.giga_tool import giga_tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import time
import sys
import json

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

class GigaChatFunctionsAgent:
    """Агент с использованием GigaChat function calling."""
    
    def __init__(self):
        """Инициализация агента."""
        self.setup_llm()
        self.setup_embeddings()
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
        
    def setup_embeddings(self):
        """Настройка embeddings для векторного поиска."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/rubert-tiny2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embeddings инициализированы")
        
    def setup_vector_stores(self):
        """Настройка векторных хранилищ."""
        persist_dir = "./vector_stores"
        os.makedirs(persist_dir, exist_ok=True)
        
        # Хранилище для документов DAMA DMBOK
        self.dama_store = Chroma(
            collection_name="dama_dmbok",
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        
        # Хранилище для методологических материалов ЦТК
        self.ctk_store = Chroma(
            collection_name="ctk_methodology",
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        
        logger.info("Векторные хранилища инициализированы")
    
    def setup_functions(self):
        """Настройка функций с использованием giga_tool декоратора."""
        
        # Примеры использования для DAMA поиска
        dama_few_shot_examples = [
            {
                "request": "Найди информацию о методологии управления данными",
                "params": {"query": "методология управления данными"}
            },
            {
                "request": "Что такое DMBOK?",
                "params": {"query": "DMBOK Data Management Body Of Knowledge"}
            },
            {
                "request": "Расскажи о ролях в управлении данными",
                "params": {"query": "роли ответственность управление данными"}
            }
        ]
        
        # Примеры использования для ЦТК поиска
        ctk_few_shot_examples = [
            {
                "request": "Найди информацию о технологических решениях",
                "params": {"query": "технологические решения архитектура"}
            },
            {
                "request": "Расскажи о методологии разработки",
                "params": {"query": "методология разработки практики"}
            },
            {
                "request": "Что такое ЦТК?",
                "params": {"query": "ЦТК технологический консалтинг"}
            }
        ]
        
        # Создаем функции с декоратором giga_tool
        @giga_tool(few_shot_examples=dama_few_shot_examples)
        def dama_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в документах DAMA DMBOK")) -> DamaSearchResult:
            """Поиск информации в документах DAMA DMBOK. Используй для поиска информации о методологии управления данными, стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными, Data Management Body Of Knowledge (DMBOK)."""
            try:
                logger.info(f"Поиск в DAMA DMBOK: {query}")
                docs = self.dama_store.similarity_search(query, k=5)
                
                if not docs:
                    return DamaSearchResult(
                        content="Информация по данному запросу не найдена в документах DAMA DMBOK.",
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
                    content=f"Ошибка при поиске в документах DAMA DMBOK: {str(e)}",
                    sources=[]
                )
        
        @giga_tool(few_shot_examples=ctk_few_shot_examples)
        def ctk_search(query: str = Field(description="Поисковый запрос на русском языке для поиска в методологических материалах ЦТК")) -> CtkSearchResult:
            """Поиск информации в методологических материалах ЦТК. Используй для поиска информации о технологических решениях, архитектуре систем, методологиях разработки, стандартах и практиках ЦТК, методологических материалах и презентациях."""
            try:
                logger.info(f"Поиск в ЦТК: {query}")
                docs = self.ctk_store.similarity_search(query, k=5)
                
                if not docs:
                    return CtkSearchResult(
                        content="Информация по данному запросу не найдена в методологических материалах ЦТК.",
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
        """Настройка агента с использованием create_react_agent."""
        try:
            # Привязываем функции к LLM
            self.llm_with_functions = self.llm.bind_functions(self.functions)
            
            # Создаем агента с памятью
            self.agent_executor = create_react_agent(
                self.llm_with_functions, 
                self.functions, 
                checkpointer=MemorySaver(),
                state_modifier="""Ты - эксперт по управлению данными. У тебя есть доступ к двум источникам информации:
1. Документы DAMA DMBOK - для информации о методологии управления данными
2. Методологические материалы ЦТК - для информации о технологических решениях

Используй соответствующие функции для поиска информации и дай подробный, структурированный ответ на русском языке."""
            )
            
            logger.info("Агент с функциями создан")
            
        except Exception as e:
            logger.error(f"Ошибка создания агента: {e}")
            raise
    
    def process_query(self, user_query: str, thread_id: str = "default") -> str:
        """
        Обработка запроса пользователя с использованием агента.
        
        Args:
            user_query: Запрос пользователя
            thread_id: ID потока для памяти
            
        Returns:
            Ответ агента
        """
        try:
            logger.info(f"Обработка запроса: {user_query}")
            
            # Конфигурация для памяти
            config = {"configurable": {"thread_id": thread_id}}
            
            # Отправляем запрос агенту
            response = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=user_query)]}, 
                config=config
            )
            
            # Получаем последнее сообщение от агента
            bot_answer = response['messages'][-1].content
            
            logger.info("Запрос обработан успешно")
            return bot_answer
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            
            # Fallback к простому запросу
            try:
                logger.info("Используем fallback - простой запрос к LLM")
                response = self.llm.invoke(user_query)
                return response.content
            except Exception as fallback_error:
                logger.error(f"Ошибка fallback: {fallback_error}")
                return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
    
    def get_store_info(self) -> Dict[str, Any]:
        """Получение информации о векторных хранилищах."""
        try:
            dama_count = self.dama_store._collection.count()
            ctk_count = self.ctk_store._collection.count()
            
            return {
                "dama_dmbok": {
                    "documents": dama_count,
                    "status": "ready" if dama_count > 0 else "empty"
                },
                "ctk_methodology": {
                    "documents": ctk_count,
                    "status": "ready" if ctk_count > 0 else "empty"
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
    """Главная функция для запуска GigaChat Functions Agent."""
    try:
        print("🚀 Инициализация GigaChat Functions Agent...")
        agent = GigaChatFunctionsAgent()
        
        # Проверяем состояние хранилищ
        store_info = agent.get_store_info()
        print("\n📊 Состояние векторных хранилищ:")
        for store_name, info in store_info.items():
            if "error" not in info:
                status_icon = "✅" if info["status"] == "ready" else "⚠️"
                print(f"   {status_icon} {store_name}: {info['documents']} документов")
            else:
                print(f"   ❌ {store_name}: ошибка - {info['error']}")
        
        # Показываем информацию о функциях
        functions_info = agent.get_functions_info()
        print(f"\n🔧 Доступные функции: {functions_info['total_functions']}")
        for func_info in functions_info['functions']:
            print(f"   - {func_info['name']}: {func_info['description'][:50]}...")
        
        print("\n" + "=" * 50)
        print("✅ GigaChat Functions Agent готов к работе!")
        print("Введите 'exit', 'quit' или 'выход' для завершения")
        print("Или нажмите Ctrl+C для принудительной остановки")
        print("=" * 50)
        
        # Функция для чата
        def chat(agent_executor, thread_id: str):
            config = {"configurable": {"thread_id": thread_id}}
            
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
                    
                    # Обработка запроса
                    start_time = time.time()
                    resp = agent_executor.invoke(
                        {"messages": [HumanMessage(content=user_input)]}, 
                        config=config
                    )
                    end_time = time.time()
                    
                    bot_answer = resp['messages'][-1].content
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
        
        # Запускаем чат
        chat(agent.agent_executor, "main_thread")
                
    except Exception as e:
        print(f"❌ Критическая ошибка инициализации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 