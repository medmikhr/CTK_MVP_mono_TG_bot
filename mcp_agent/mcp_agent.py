#!/usr/bin/env python3
"""
MCP-совместимый агент для работы с документами DAMA DMBOK и методологическими материалами ЦТК
"""

import os
import logging
from typing import List, Dict, Any, Callable, Optional
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_chroma import Chroma
from langchain.agents import tool
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
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

class MCPTool:
    """Базовый класс для MCP-совместимых инструментов."""
    
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
        
    def invoke(self, **kwargs) -> Any:
        """Вызов инструмента."""
        return self.function(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для MCP."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    }
                },
                "required": ["query"]
            }
        }

class MCPToolRegistry:
    """Реестр MCP-совместимых инструментов."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        
    def register_tool(self, tool: MCPTool):
        """Регистрация инструмента."""
        self.tools[tool.name] = tool
        logger.info(f"Зарегистрирован инструмент: {tool.name}")
        
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Получение инструмента по имени."""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """Список всех инструментов."""
        return list(self.tools.keys())
        
    def invoke_tool(self, name: str, **kwargs) -> Any:
        """Вызов инструмента по имени."""
        tool = self.get_tool(name)
        if tool:
            return tool.invoke(**kwargs)
        else:
            raise ValueError(f"Инструмент {name} не найден")

class MCPDataManagementAgent:
    """MCP-совместимый агент для работы с документами по управлению данными."""
    
    def __init__(self):
        """Инициализация агента."""
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vector_stores()
        self.setup_tool_registry()
        self.register_local_tools()
        
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
        
    def setup_tool_registry(self):
        """Настройка реестра инструментов."""
        self.tool_registry = MCPToolRegistry()
        logger.info("Реестр инструментов инициализирован")
        
    def register_local_tools(self):
        """Регистрация локальных инструментов."""
        
        # Инструмент поиска в DAMA DMBOK
        def dama_search(query: str) -> str:
            try:
                logger.info(f"Поиск в DAMA DMBOK: {query}")
                docs = self.dama_store.similarity_search(query, k=5)
                
                if not docs:
                    return "Информация по данному запросу не найдена в документах DAMA DMBOK."
                
                result = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Неизвестный источник')
                    result.append(f"Источник {i}: {source}\n{doc.page_content}")
                
                return "\n\n---\n\n".join(result)
                
            except Exception as e:
                logger.error(f"Ошибка поиска в DAMA: {e}")
                return f"Ошибка при поиске в документах DAMA DMBOK: {str(e)}"
        
        dama_tool = MCPTool(
            name="dama_search",
            description="Поиск информации в документах DAMA DMBOK. Используй для поиска информации о методологии управления данными, стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными, Data Management Body Of Knowledge (DMBOK).",
            function=dama_search
        )
        self.tool_registry.register_tool(dama_tool)
        
        # Инструмент поиска в ЦТК
        def ctk_search(query: str) -> str:
            try:
                logger.info(f"Поиск в ЦТК: {query}")
                docs = self.ctk_store.similarity_search(query, k=5)
                
                if not docs:
                    return "Информация по данному запросу не найдена в методологических материалах ЦТК."
                
                result = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Неизвестный источник')
                    result.append(f"Источник {i}: {source}\n{doc.page_content}")
                
                return "\n\n---\n\n".join(result)
                
            except Exception as e:
                logger.error(f"Ошибка поиска в ЦТК: {e}")
                return f"Ошибка при поиске в материалах ЦТК: {str(e)}"
        
        ctk_tool = MCPTool(
            name="ctk_search",
            description="Поиск информации в методологических материалах ЦТК. Используй для поиска информации о технологических решениях, архитектуре систем, методологиях разработки, стандартах и практиках ЦТК, методологических материалах и презентациях.",
            function=ctk_search
        )
        self.tool_registry.register_tool(ctk_tool)
        
        logger.info("Локальные инструменты зарегистрированы")
    
    def add_mcp_tool(self, name: str, description: str, function: Callable):
        """Добавление MCP инструмента."""
        tool = MCPTool(name=name, description=description, function=function)
        self.tool_registry.register_tool(tool)
        logger.info(f"Добавлен MCP инструмент: {name}")
    
    def remove_tool(self, name: str):
        """Удаление инструмента."""
        if name in self.tool_registry.tools:
            del self.tool_registry.tools[name]
            logger.info(f"Удален инструмент: {name}")
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Список доступных инструментов в MCP формате."""
        return [tool.to_dict() for tool in self.tool_registry.tools.values()]
    
    def select_tools_for_query(self, user_query: str) -> List[str]:
        """Выбор инструментов для запроса на основе ключевых слов."""
        tools_to_use = []
        
        # Ключевые слова для DAMA
        dama_keywords = [
            'dama', 'dmbok', 'управление данными', 'методология', 
            'стандарты', 'процессы', 'роли', 'ответственность',
            'data governance', 'data management'
        ]
        
        # Ключевые слова для ЦТК
        ctk_keywords = [
            'цтк', 'технологии', 'архитектура', 'разработка', 
            'системы', 'методология', 'практики', 'решения',
            'технологический консалтинг'
        ]
        
        user_query_lower = user_query.lower()
        
        # Выбираем инструменты на основе ключевых слов
        if any(keyword in user_query_lower for keyword in dama_keywords):
            if 'dama_search' in self.tool_registry.tools:
                tools_to_use.append('dama_search')
                
        if any(keyword in user_query_lower for keyword in ctk_keywords):
            if 'ctk_search' in self.tool_registry.tools:
                tools_to_use.append('ctk_search')
        
        # Если ключевые слова не найдены, используем все доступные инструменты
        if not tools_to_use:
            tools_to_use = list(self.tool_registry.tools.keys())
        
        return tools_to_use
    
    def process_query(self, user_query: str) -> str:
        """
        Обработка запроса пользователя с использованием MCP-совместимых инструментов.
        
        Args:
            user_query: Запрос пользователя
            
        Returns:
            Ответ агента
        """
        try:
            logger.info(f"Обработка запроса: {user_query}")
            
            # Выбираем инструменты для запроса
            tools_to_use = self.select_tools_for_query(user_query)
            logger.info(f"Выбраны инструменты: {tools_to_use}")
            
            # Собираем информацию из инструментов
            collected_info = []
            
            for tool_name in tools_to_use:
                logger.info(f"Используем инструмент: {tool_name}")
                try:
                    result = self.tool_registry.invoke_tool(tool_name, query=user_query)
                    if result and len(str(result).strip()) > 0:
                        collected_info.append(f"=== Информация из {tool_name} ===\n{result}")
                    else:
                        logger.warning(f"Пустой результат от {tool_name}")
                except Exception as e:
                    logger.error(f"Ошибка инструмента {tool_name}: {e}")
            
            # Формируем ответ
            if collected_info:
                context = "\n\n".join(collected_info)
                
                prompt = f"""Ты - эксперт по управлению данными. На основе предоставленной информации ответь на вопрос пользователя.

Контекст:
{context}

Вопрос пользователя: {user_query}

Инструкции:
1. Ответь подробно и структурированно
2. Используй информацию из контекста
3. Если в контексте нет информации для ответа, скажи об этом честно
4. Отвечай на русском языке
5. Структурируй ответ с использованием заголовков и списков

Ответ:"""
                
                logger.info("Отправляем запрос к LLM")
                response = self.llm.invoke(prompt)
                return response.content
            else:
                # Fallback к простому запросу
                logger.info("Используем fallback - простой запрос к LLM")
                response = self.llm.invoke(user_query)
                return response.content
                
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
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
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Получение информации о доступных инструментах."""
        return {
            "total_tools": len(self.tool_registry.tools),
            "tool_names": list(self.tool_registry.tools.keys()),
            "mcp_format": self.list_available_tools()
        }

def main():
    """Главная функция для запуска MCP-совместимого агента."""
    try:
        print("🚀 Инициализация MCP-совместимого агента управления данными...")
        agent = MCPDataManagementAgent()
        
        # Проверяем состояние хранилищ
        store_info = agent.get_store_info()
        print("\n📊 Состояние векторных хранилищ:")
        for store_name, info in store_info.items():
            if "error" not in info:
                status_icon = "✅" if info["status"] == "ready" else "⚠️"
                print(f"   {status_icon} {store_name}: {info['documents']} документов")
            else:
                print(f"   ❌ {store_name}: ошибка - {info['error']}")
        
        # Показываем информацию об инструментах
        tool_info = agent.get_tool_info()
        print(f"\n🔧 Доступные инструменты: {tool_info['total_tools']}")
        for tool_name in tool_info['tool_names']:
            print(f"   - {tool_name}")
        
        print("\n" + "=" * 50)
        print("✅ MCP-совместимый агент готов к работе!")
        print("Введите 'exit', 'quit' или 'выход' для завершения")
        print("Или нажмите Ctrl+C для принудительной остановки")
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
                
                # Обработка запроса
                start_time = time.time()
                response = agent.process_query(user_input)
                end_time = time.time()
                
                print(f"\n💬 Ответ (за {end_time - start_time:.2f}с):")
                print(f"{response}")
                
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