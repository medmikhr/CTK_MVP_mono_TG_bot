import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import tool
from document_processor_langchain import PERSIST_DIR
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

# GIGACHAT_TOKEN = N2ZiYzFhNDktYzMzOC00ZGEwLTg2ODktN2U1OTZlNjNmNTZiOjY1M2U0ZmY0LTAxOTYtNGFhZS1hMjBhLThhNTIzMTczNGZhZQ== Дмитрий
# GIGACHAT_TOKEN = NjVlYWZhODAtZmYwZC00ODUwLTgwMDQtOGUwZjc0OWM1MDJjOmRjZjNhYWJkLTM3OTQtNDRlMC1hZjBkLTNiMmZlNTgzNTg1NA== Сергей
# GIGACHAT_TOKEN = M2Y1Y2VhYzItZGQ1ZS00MWI5LWFiMDMtY2JmNzFkNDY1N2RiOjU1YmJhNDRjLTRlYTAtNGYzZC04ZDdmLWE3NzBmNzZkNjA2Mg== Денис

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

tools_dict = [dama_retrieve_tool, ctk_retrieve_tool, sbf_retrieve_tool]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools_dict, checkpointer=memory)

def agent_ask(user_id, input_message):
    """Функция для обработки запросов пользователя с обработкой ошибок."""
    try:
        config = {"configurable": {"thread_id": user_id}}
        event_count = 0
        
        print(f"\n🔍 Обработка запроса: '{input_message}'")
        print("=" * 50)
        
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            event_count += 1
            print(f"\n📋 Событие #{event_count}:")
            
            # Выводим тип события и ключи
            print(f"   Тип события: {type(event).__name__}")
            print(f"   Ключи события: {list(event.keys())}")
            
            # Обрабатываем сообщения
            if "messages" in event and event["messages"]:
                messages = event["messages"]
                print(f"   Количество сообщений: {len(messages)}")
                
                for i, message in enumerate(messages):
                    print(f"   Сообщение {i+1}:")
                    print(f"      Тип: {type(message).__name__}")
                    
                    if hasattr(message, 'content'):
                        content = message.content
                        print(f"      Контент: {content[:200]}...")
                        
                        # Если это сообщение от агента с инструментами
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            print(f"      🔧 Вызовы инструментов: {len(message.tool_calls)}")
                            for j, tool_call in enumerate(message.tool_calls):
                                print(f"         Инструмент {j+1}: {tool_call}")
                    
                    # Если есть дополнительные атрибуты
                    if hasattr(message, 'additional_kwargs'):
                        additional = message.additional_kwargs
                        if additional:
                            print(f"      Дополнительно: {additional}")
            
            # Обрабатываем tool_calls
            if "tool_calls" in event:
                tool_calls = event["tool_calls"]
                print(f"   🔧 Вызовы инструментов: {len(tool_calls)}")
                for i, tool_call in enumerate(tool_calls):
                    print(f"      Инструмент {i+1}: {tool_call}")
            
            # Обрабатываем tool_results
            if "tool_results" in event:
                tool_results = event["tool_results"]
                print(f"   📊 Результаты инструментов: {len(tool_results)}")
                for i, result in enumerate(tool_results):
                    print(f"      Результат {i+1}: {str(result)[:200]}...")
            
            # Выводим только финальное сообщение для пользователя
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content') and event_count > 1:  # Пропускаем первое событие (входной запрос)
                    print(f"\n💬 Ответ агента:")
                    print(f"{last_message.content}")
        
        # Возвращаем последнее сообщение
        if "messages" in event and event["messages"]:
            return event["messages"][-1]
        else:
            return "Нет ответа"
            
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса пользователя {user_id}: {e}")
        
        # Попытка простого запроса без инструментов
        try:
            logger.info("Попытка простого запроса без инструментов...")
            simple_response = llm.invoke(input_message)
            print(f"\n=== Простой ответ (без инструментов) ===")
            print(simple_response.content)
            return simple_response
        except Exception as simple_error:
            logger.error(f"Ошибка при простом запросе: {simple_error}")
            error_message = f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
            print(f"\n❌ {error_message}")
            return error_message

def test_vector_stores():
    """Тестирование векторных хранилищ."""
    logger.info("Тестирование векторных хранилищ...")
    
    try:
        for store_name, store in vector_stores.items():
            logger.info(f"Тестирование хранилища {store_name}...")
            
            # Проверяем количество документов
            count = store._collection.count()
            logger.info(f"Хранилище {store_name}: {count} документов")
            
            # Тестовый поиск
            if count > 0:
                test_query = "тест"
                docs = store.similarity_search(test_query, k=1)
                logger.info(f"Тестовый поиск в {store_name} успешен")
            else:
                logger.warning(f"Хранилище {store_name} пустое")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании векторных хранилищ: {e}")
        return False

def test_ctk_tool_before_startup():
    """Тестирование ctk_retrieve_tool перед запуском."""
    print("\n🔍 Тестирование ctk_retrieve_tool перед запуском...")
    print("=" * 50)
    
    try:
        # Проверяем состояние хранилища CTK
        ctk_store = vector_stores["ctk"]
        count = ctk_store._collection.count()
        print(f"Хранилище CTK: {count} документов")
        
        if count == 0:
            print("⚠️  Хранилище CTK пустое - инструмент может не работать")
            return False
        
        # Тестовые запросы
        test_queries = [
            "слои информационной архитектуры",
            "технологические решения"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Тест #{i}: '{query}'")
            print("-" * 30)
            
            start_time = time.time()
            
            try:
                # Прямой вызов инструмента
                result = ctk_retrieve_tool.invoke(query)
                end_time = time.time()
                
                print(f"✅ Прямой вызов: {end_time - start_time:.2f}с")
                print(f"📄 Тип результата: {type(result).__name__}")
                
                # Анализ результата
                if isinstance(result, str):
                    print(f"📝 Результат: {len(result)} символов")
                    print(f"📄 Содержимое: {result[:200]}...")
                elif isinstance(result, tuple):
                    print(f"📝 Кортеж: {len(result)} элементов")
                    for j, item in enumerate(result):
                        print(f"   Элемент {j+1}: {str(item)[:100]}...")
                else:
                    print(f"📝 Результат: {str(result)[:200]}...")
                
                # Тест через агента
                print(f"\n🔍 Тест через агента...")
                agent_start_time = time.time()
                
                config = {"configurable": {"thread_id": f"startup_test_{i}"}}
                event_count = 0
                
                for event in agent_executor.stream(
                    {"messages": [{"role": "user", "content": f"Используй ctk_retrieve_tool для поиска информации о {query}"}]},
                    stream_mode="values",
                    config=config,
                ):
                    event_count += 1
                    print(f"   📋 Событие #{event_count}:")
                    
                    # Выводим тип события
                    if "messages" in event:
                        messages = event["messages"]
                        if messages:
                            last_message = messages[-1]
                            print(f"      Тип: {type(last_message).__name__}")
                            
                            # Если это сообщение от агента
                            if hasattr(last_message, 'content'):
                                print(f"      Контент: {last_message.content[:100]}...")
                    
                    # Выводим все ключи события для отладки
                    print(f"      Ключи события: {list(event.keys())}")
                    
                    # Если есть tool_calls
                    if "tool_calls" in event:
                        tool_calls = event["tool_calls"]
                        print(f"      🔧 Вызовы инструментов: {len(tool_calls)}")
                        for j, tool_call in enumerate(tool_calls):
                            print(f"         Инструмент {j+1}: {tool_call}")
                    
                    # Если есть tool_results
                    if "tool_results" in event:
                        tool_results = event["tool_results"]
                        print(f"      📊 Результаты инструментов: {len(tool_results)}")
                        for j, result in enumerate(tool_results):
                            print(f"         Результат {j+1}: {str(result)[:100]}...")
                
                agent_end_time = time.time()
                print(f"✅ Агент: {agent_end_time - agent_start_time:.2f}с ({event_count} событий)")
                
            except Exception as e:
                end_time = time.time()
                print(f"❌ Ошибка за {end_time - start_time:.2f}с: {e}")
                print(f"   Тип ошибки: {type(e).__name__}")
                return False
        
        print(f"\n✅ Тестирование ctk_retrieve_tool завершено успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        return False

def test_gigachat_connection():
    """Тестирование подключения к GigaChat."""
    try:
        # Простой тест LLM
        test_response = llm.invoke("Привет! Как дела?")
        logger.info("Тест подключения к GigaChat успешен")
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании GigaChat: {e}")
        return False

def test_agent_tools_registration():
    """Проверка регистрации инструментов в агенте."""
    print("\n🔧 Проверка регистрации инструментов в агенте...")
    print("=" * 50)
    
    try:
        # Получаем список доступных инструментов
        available_tools = agent_executor.get_tools()
        print(f"Доступные инструменты: {len(available_tools)}")
        
        for i, tool in enumerate(available_tools, 1):
            print(f"   {i}. {tool.name}: {tool.description[:100]}...")
        
        # Проверяем, что наши инструменты есть в списке
        tool_names = [tool.name for tool in available_tools]
        expected_tools = ["dama_retrieve_tool", "ctk_retrieve_tool", "sbf_retrieve_tool"]
        
        missing_tools = [tool for tool in expected_tools if tool not in tool_names]
        
        if missing_tools:
            print(f"❌ Отсутствующие инструменты: {missing_tools}")
            return False
        else:
            print("✅ Все инструменты зарегистрированы")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка при проверке инструментов: {e}")
        return False

if __name__ == '__main__':
    # Проверяем флаги запуска
    skip_tests = "--skip-tests" in sys.argv or "--fast" in sys.argv
    
    if skip_tests:
        print("⚡ Быстрый запуск (тестирование пропущено)")
    else:
        print("🔍 Запуск с полным тестированием")
    
    # Тестирование подключения к GigaChat
    if not test_gigachat_connection():
        print("❌ Ошибка при подключении к GigaChat")
        print("Проверьте токен и сетевое подключение")
        exit(1)
    
    # Тестирование векторных хранилищ
    if not test_vector_stores():
        print("⚠️  Проблемы с векторными хранилищами")
        print("Возможно, хранилища пустые или повреждены")
        print("Попробуйте загрузить документы снова")
    
    # Тестирование ctk_retrieve_tool (если не пропущено)
    if not skip_tests:
        if not test_ctk_tool_before_startup():
            print("⚠️  Проблемы с ctk_retrieve_tool")
            print("Инструмент может не работать корректно")
            print("Рекомендуется использовать agent_simple.py")
    else:
        print("⏭️  Тестирование ctk_retrieve_tool пропущено")
    
    # Тестирование регистрации инструментов
    if not test_agent_tools_registration():
        print("⚠️  Проблемы с регистрацией инструментов")
        print("Инструменты не зарегистрированы корректно")
        print("Рекомендуется использовать agent_simple.py")
    
    print("\n" + "=" * 50)
    print("✅ Агент готов к работе!")
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
            
            agent_ask(1, user_input)
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем (Ctrl+C)")
        print("До свидания!")
    except EOFError:
        print("\n\nПрограмма завершена (EOF)")
        print("До свидания!")
