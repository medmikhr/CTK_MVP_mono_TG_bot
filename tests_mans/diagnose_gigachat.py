#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с GigaChat API
"""

import os
import logging
import time
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Проверка переменных окружения."""
    print("🔍 Проверка переменных окружения...")
    
    gc_token = os.getenv('GIGACHAT_TOKEN')
    if gc_token:
        print("✅ GIGACHAT_TOKEN найден")
        print(f"   Длина токена: {len(gc_token)} символов")
        print(f"   Начинается с: {gc_token[:10]}...")
    else:
        print("❌ GIGACHAT_TOKEN не найден")
        return False
    
    return True

def test_gigachat_connection():
    """Тестирование подключения к GigaChat."""
    print("\n🔍 Тестирование подключения к GigaChat...")
    
    try:
        # Получение токена
        gc_token = os.getenv('GIGACHAT_TOKEN')
        if not gc_token:
            print("❌ Токен не найден")
            return False
        
        # Инициализация GigaChat
        print("   Инициализация GigaChat...")
        llm = GigaChat(
            credentials=gc_token,
            model='GigaChat:latest',
            verify_ssl_certs=False,
            profanity_check=False
        )
        
        # Простой тест
        print("   Отправка тестового запроса...")
        start_time = time.time()
        response = llm.invoke("Привет! Как дела?")
        end_time = time.time()
        
        print(f"✅ Подключение успешно!")
        print(f"   Время ответа: {end_time - start_time:.2f} секунд")
        print(f"   Ответ: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_gigachat_with_tools():
    """Тестирование GigaChat с реальными инструментами."""
    print("\n🔍 Тестирование GigaChat с реальными инструментами...")
    
    try:
        gc_token = os.getenv('GIGACHAT_TOKEN')
        llm = GigaChat(
            credentials=gc_token,
            model='GigaChat:latest',
            verify_ssl_certs=False,
            profanity_check=False
        )
        
        # Создаем простой тестовый инструмент
        @tool
        def test_tool(query: str):
            """Тестовый инструмент для проверки работы с инструментами."""
            return f"Тестовый ответ на запрос: {query}"
        
        # Создаем агента с инструментом
        print("   Создание агента с инструментами...")
        tools = [test_tool]
        memory = MemorySaver()
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)
        
        # Тестируем агента
        test_query = "Используй test_tool для получения информации о слоях информационной архитектуры"
        print(f"   Тестовый запрос: {test_query}")
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "test"}}
        
        # Выполняем запрос через агента
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": test_query}]},
            stream_mode="values",
            config=config,
        ):
            answer_message = event["messages"][-1]
            print(f"   Получен ответ от агента")
        
        end_time = time.time()
        
        print(f"✅ Запрос с инструментами обработан успешно!")
        print(f"   Время ответа: {end_time - start_time:.2f} секунд")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании с инструментами: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        return False

def check_network_connectivity_alternative():
    """Альтернативная проверка сетевого подключения через socket."""
    print("\n🔍 Альтернативная проверка сети (через socket)...")
    
    import socket
    
    try:
        # Создаем socket соединение
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # Пытаемся подключиться к порту 443 (HTTPS)
        result = sock.connect_ex(('gigachat.devices.sberbank.ru', 443))
        sock.close()
        
        if result == 0:
            print("✅ Сетевое подключение работает (порт 443 доступен)")
            return True
        else:
            print(f"❌ Порт 443 недоступен (код ошибки: {result})")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка сетевого подключения: {e}")
        return False

def check_network_connectivity():
    """Проверка сетевого подключения."""
    print("\n🔍 Проверка сетевого подключения...")
    
    import urllib.request
    import socket
    import ssl
    
    # Проверка DNS
    try:
        socket.gethostbyname("gigachat.devices.sberbank.ru")
        print("✅ DNS резолвинг работает")
    except Exception as e:
        print(f"❌ Проблема с DNS: {e}")
        return False
    
    # Проверка HTTPS подключения (без проверки SSL сертификатов)
    try:
        # Создаем контекст SSL без проверки сертификатов
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Создаем opener с нашим SSL контекстом
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        
        # Тестируем подключение
        response = opener.open("https://gigachat.devices.sberbank.ru", timeout=10)
        print("✅ HTTPS подключение к GigaChat работает")
        return True
        
    except Exception as e:
        print(f"❌ Проблема с HTTPS: {e}")
        print("   (Это может быть нормально в корпоративных сетях)")
        # Попробуем альтернативную проверку
        return check_network_connectivity_alternative()

def test_real_tools():
    """Тестирование реальных инструментов с векторными хранилищами."""
    print("\n🔍 Тестирование реальных инструментов...")
    
    try:
        from document_processor_langchain import PERSIST_DIR
        from embeddings_manager import get_local_huggingface_embeddings
        from langchain_chroma import Chroma
        
        # Инициализация embeddings
        print("   Инициализация embeddings...")
        embeddings = get_local_huggingface_embeddings()
        
        # Инициализация векторных хранилищ
        print("   Инициализация векторных хранилищ...")
        vector_stores = {
            "ctk": Chroma(collection_name="ctk", persist_directory=PERSIST_DIR, embedding_function=embeddings),
        }
        
        # Проверка хранилища
        store = vector_stores["ctk"]
        count = store._collection.count()
        print(f"   Хранилище CTK: {count} документов")
        
        if count == 0:
            print("   ⚠️  Хранилище пустое - загрузите документы")
            return False
        
        # Тестовый поиск
        print("   Тестовый поиск в хранилище...")
        docs = store.similarity_search("информационная архитектура", k=1)
        print(f"   ✅ Найдено {len(docs)} документов")
        
        # Тест инструмента
        print("   Тестирование инструмента ctk_retrieve_tool...")
        from langchain.agents import tool
        
        @tool
        def test_ctk_tool(query: str):
            """Тестовый инструмент для проверки CTK хранилища."""
            docs = store.similarity_search(query, k=2)
            return f"Найдено {len(docs)} документов для запроса: {query}"
        
        # Создаем агента с реальным инструментом
        from langchain_gigachat import GigaChat
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        
        gc_token = os.getenv('GIGACHAT_TOKEN')
        llm = GigaChat(
            credentials=gc_token,
            model='GigaChat:latest',
            verify_ssl_certs=False,
            profanity_check=False
        )
        
        tools = [test_ctk_tool]
        memory = MemorySaver()
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)
        
        # Тестируем агента
        test_query = "Используй test_ctk_tool для поиска информации о слоях информационной архитектуры"
        print(f"   Тестовый запрос: {test_query}")
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "test_real"}}
        
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": test_query}]},
            stream_mode="values",
            config=config,
        ):
            answer_message = event["messages"][-1]
            print(f"   Получен ответ от агента с реальным инструментом")
        
        end_time = time.time()
        
        print(f"✅ Реальные инструменты работают!")
        print(f"   Время ответа: {end_time - start_time:.2f} секунд")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании реальных инструментов: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        return False

def test_real_ctk_tool():
    """Тестирование реальной ctk_retrieve_tool из agent.py."""
    print("\n🔍 Тестирование реальной ctk_retrieve_tool...")
    
    try:
        # Импортируем реальные компоненты из agent.py
        from old_react_agent import ctk_retrieve_tool, vector_stores
        
        print("   Импорт реальных компонентов из agent.py...")
        
        # Проверяем состояние хранилища CTK
        ctk_store = vector_stores["ctk"]
        count = ctk_store._collection.count()
        print(f"   Хранилище CTK: {count} документов")
        
        if count == 0:
            print("   ⚠️  Хранилище CTK пустое")
            return False
        
        # Тестируем реальный инструмент напрямую
        print("   Тестирование ctk_retrieve_tool напрямую...")
        test_query = "слои информационной архитектуры"
        
        start_time = time.time()
        result = ctk_retrieve_tool.invoke(test_query)
        end_time = time.time()
        
        print(f"   ✅ ctk_retrieve_tool работает!")
        print(f"   Время выполнения: {end_time - start_time:.2f} секунд")
        print(f"   Результат: {str(result)[:200]}...")
        
        # Тестируем через LangGraph агента
        print("   Тестирование через LangGraph агента...")
        from old_react_agent import agent_executor
        
        start_time = time.time()
        config = {"configurable": {"thread_id": "test_ctk_real"}}
        
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": f"Используй ctk_retrieve_tool для поиска информации о {test_query}"}]},
            stream_mode="values",
            config=config,
        ):
            answer_message = event["messages"][-1]
            print(f"   Получен ответ от агента с реальной ctk_retrieve_tool")
        
        end_time = time.time()
        
        print(f"✅ Реальная ctk_retrieve_tool работает через агента!")
        print(f"   Время ответа агента: {end_time - start_time:.2f} секунд")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании реальной ctk_retrieve_tool: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        return False

def main():
    """Главная функция диагностики."""
    print("🚀 Диагностика GigaChat API")
    print("=" * 50)
    
    results = {}
    
    # Проверка окружения
    print("\n1️⃣ Проверка переменных окружения...")
    results['environment'] = check_environment()
    
    # Проверка сети
    print("\n2️⃣ Проверка сетевого подключения...")
    results['network'] = check_network_connectivity()
    
    # Тест подключения к GigaChat
    print("\n3️⃣ Тест подключения к GigaChat...")
    results['connection'] = test_gigachat_connection()
    
    # Тест с инструментами
    print("\n4️⃣ Тест с инструментами...")
    results['tools'] = test_gigachat_with_tools()
    
    # Тест реальных инструментов
    print("\n5️⃣ Тест реальных инструментов...")
    results['real_tools'] = test_real_tools()
    
    # Тест реальной ctk_retrieve_tool
    print("\n6️⃣ Тест реальной ctk_retrieve_tool...")
    results['real_ctk_tool'] = test_real_ctk_tool()
    
    # Итоговый отчет
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    print(f"Переменные окружения: {'✅' if results['environment'] else '❌'}")
    print(f"Сетевое подключение: {'✅' if results['network'] else '❌'}")
    print(f"Подключение к GigaChat: {'✅' if results['connection'] else '❌'}")
    print(f"Работа с инструментами: {'✅' if results['tools'] else '❌'}")
    print(f"Реальные инструменты: {'✅' if results['real_tools'] else '❌'}")
    print(f"Реальная ctk_retrieve_tool: {'✅' if results['real_ctk_tool'] else '❌'}")
    
    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    
    if not results['environment']:
        print("❌ Проверьте файл .env и переменную GIGACHAT_TOKEN")
    
    if not results['network']:
        print("❌ Проверьте интернет-соединение и доступность GigaChat API")
        print("   Примечание: В корпоративных сетях SSL проверка может не проходить")
        print("   Если GigaChat работает, то это не критично")
    
    if not results['connection']:
        print("❌ Проблемы с подключением к GigaChat - проверьте токен")
    
    if not results['tools']:
        if results['connection']:
            print("⚠️  Проблемы с инструментами, но базовое подключение работает")
            print("   Рекомендуется использовать agent_simple.py")
        else:
            print("❌ Не удалось протестировать инструменты из-за проблем с подключением")
    
    if not results['real_tools']:
        if results['connection'] and results['tools']:
            print("⚠️  Проблемы с реальными инструментами, но базовое подключение работает")
            print("   Рекомендуется использовать agent_simple.py")
        else:
            print("❌ Не удалось протестировать реальные инструменты")
    
    if not results['real_ctk_tool']:
        if results['connection'] and results['tools'] and results['real_tools']:
            print("⚠️  Проблемы с реальной ctk_retrieve_tool, но базовое подключение работает")
            print("   Рекомендуется использовать agent_simple.py")
        else:
            print("❌ Не удалось протестировать реальную ctk_retrieve_tool")
    
    if all(results.values()):
        print("✅ Все тесты пройдены успешно!")
        print("   Можно использовать любой из агентов:")
        print("   - agent.py (полный функционал)")
        print("   - agent_gigachat.py (с GigaChat embeddings)")
        print("   - agent_simple.py (простой режим)")
    elif results['connection'] and results['tools'] and results['real_tools'] and results['real_ctk_tool']:
        print("✅ Основные компоненты работают!")
        print("   Можно использовать любой из агентов")
        print("   (Проблемы с сетевой диагностикой не критичны)")
    
    print("\n" + "=" * 50)
    print("Диагностика завершена")

if __name__ == "__main__":
    main() 