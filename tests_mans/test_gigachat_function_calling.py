#!/usr/bin/env python3
"""
Тест GigaChat Function Calling - проверка реализации согласно документации Сбера
https://developers.sber.ru/docs/ru/gigachat/guides/function-calling
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_gigachat.tools.giga_tool import giga_tool
from pydantic import BaseModel, Field
import time

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Тестовые модели
class WeatherResult(BaseModel):
    """Результат получения погоды."""
    temperature: str = Field(description="Температура")
    condition: str = Field(description="Состояние погоды")
    location: str = Field(description="Местоположение")

class CalculatorResult(BaseModel):
    """Результат вычисления."""
    result: float = Field(description="Результат вычисления")
    operation: str = Field(description="Выполненная операция")

def test_gigachat_function_calling():
    """Тест GigaChat function calling согласно документации Сбера."""
    
    print("🧪 Тестирование GigaChat Function Calling")
    print("=" * 50)
    
    try:
        # 1. Инициализация GigaChat
        print("\n1️⃣ Инициализация GigaChat...")
        gc_auth = os.getenv('GIGACHAT_TOKEN')
        if not gc_auth:
            raise ValueError("Не найден токен GigaChat")
        
        llm = GigaChat(
            credentials=gc_auth,
            model='GigaChat:latest',
            verify_ssl_certs=False,
            profanity_check=False
        )
        print("✅ GigaChat инициализирован")
        
        # 2. Создание функций с giga_tool
        print("\n2️⃣ Создание функций с giga_tool...")
        
        # Примеры для функции погоды
        weather_examples = [
            {
                "request": "Какая погода в Москве?",
                "params": {"location": "Москва"}
            },
            {
                "request": "Погода в Санкт-Петербурге",
                "params": {"location": "Санкт-Петербург"}
            }
        ]
        
        # Примеры для калькулятора
        calculator_examples = [
            {
                "request": "Сколько будет 2 + 2?",
                "params": {"a": 2, "b": 2, "operation": "add"}
            },
            {
                "request": "Умножь 5 на 3",
                "params": {"a": 5, "b": 3, "operation": "multiply"}
            }
        ]
        
        @giga_tool(few_shot_examples=weather_examples)
        def get_weather(location: str = Field(description="Город для получения погоды")) -> WeatherResult:
            """Получение информации о погоде в указанном городе."""
            # Имитация получения погоды
            return WeatherResult(
                temperature="20°C",
                condition="Солнечно",
                location=location
            )
        
        @giga_tool(few_shot_examples=calculator_examples)
        def calculate(a: float = Field(description="Первое число"), 
                     b: float = Field(description="Второе число"),
                     operation: str = Field(description="Операция: add, subtract, multiply, divide")) -> CalculatorResult:
            """Выполнение математических операций."""
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                result = a / b if b != 0 else 0
            else:
                result = 0
            
            return CalculatorResult(
                result=result,
                operation=operation
            )
        
        print("✅ Функции созданы с giga_tool")
        
        # 3. Привязка функций к LLM
        print("\n3️⃣ Привязка функций к LLM...")
        llm_with_functions = llm.bind_tools([get_weather, calculate])
        print("✅ Функции привязаны к LLM")
        
        # 4. Тестирование вызовов
        print("\n4️⃣ Тестирование вызовов функций...")
        
        # Тест 1: Погода
        print("\n🌤️ Тест функции погоды:")
        messages = [
            SystemMessage(content="Ты - помощник с доступом к функциям получения погоды и калькулятору."),
            HumanMessage(content="Какая погода в Москве?")
        ]
        
        start_time = time.time()
        response = llm_with_functions.invoke(messages)
        end_time = time.time()
        
        print(f"⏱️ Время ответа: {end_time - start_time:.2f}с")
        print(f"🤖 Ответ: {response.content}")
        
        # Проверяем, была ли вызвана функция
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("✅ Функция была вызвана!")
            for tool_call in response.tool_calls:
                print(f"   📞 Вызвана функция: {tool_call['name']}")
                print(f"   📝 Аргументы: {tool_call['args']}")
        else:
            print("⚠️ Функция не была вызвана")
        
        # Тест 2: Калькулятор
        print("\n🧮 Тест функции калькулятора:")
        messages = [
            SystemMessage(content="Ты - помощник с доступом к функциям получения погоды и калькулятору."),
            HumanMessage(content="Сколько будет 15 умножить на 7?")
        ]
        
        start_time = time.time()
        response = llm_with_functions.invoke(messages)
        end_time = time.time()
        
        print(f"⏱️ Время ответа: {end_time - start_time:.2f}с")
        print(f"🤖 Ответ: {response.content}")
        
        # Проверяем, была ли вызвана функция
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("✅ Функция была вызвана!")
            for tool_call in response.tool_calls:
                print(f"   📞 Вызвана функция: {tool_call['name']}")
                print(f"   📝 Аргументы: {tool_call['args']}")
        else:
            print("⚠️ Функция не была вызвана")
        
        # 5. Тест с прямой обработкой tool_calls
        print("\n5️⃣ Тест с обработкой tool_calls...")
        
        def process_with_tool_calls(messages):
            """Обработка с автоматическим вызовом функций."""
            while True:
                response = llm_with_functions.invoke(messages)
                # Если есть tool_calls, обрабатываем их
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        func_name = tool_call['name']
                        args = tool_call['args']
                        print(f"🔧 Выполняю функцию {func_name} с аргументами {args}")
                        if func_name == "get_weather":
                            result = get_weather.invoke(args)
                        elif func_name == "calculate":
                            result = calculate.invoke(args)
                        else:
                            result = None
                        # Добавляем результат вызова функции в историю
                        messages.append(FunctionMessage(name=func_name, content=result.json()))
                else:
                    return response.content
        
        # Тест обработки
        print("\n🔄 Тест с автоматической обработкой:")
        messages = [
            SystemMessage(content="Ты - помощник с доступом к функциям. Используй их когда нужно."),
            HumanMessage(content="Какая погода в Санкт-Петербурге и сколько будет 10 + 5?")
        ]
        
        start_time = time.time()
        result = process_with_tool_calls(messages)
        end_time = time.time()
        
        print(f"⏱️ Время обработки: {end_time - start_time:.2f}с")
        print(f"🤖 Финальный ответ: {result}")
        
        print("\n✅ Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        logger.error(f"Ошибка тестирования: {e}")
        return False

def test_our_agent_implementation():
    """Тест нашей реализации агента."""
    print("\n" + "=" * 60)
    print("🧪 Тестирование нашей реализации агента")
    print("=" * 60)
    
    try:
        from class_functions_agent import GigaChatFunctionsAgent
        
        print("\n1️⃣ Инициализация нашего агента...")
        agent = GigaChatFunctionsAgent()
        print("✅ Агент инициализирован")
        
        print("\n2️⃣ Тест обработки запроса...")
        test_query = "Расскажи о методологии управления данными"
        
        start_time = time.time()
        response = agent.process_query(test_query, thread_id="test")
        end_time = time.time()
        
        print(f"⏱️ Время обработки: {end_time - start_time:.2f}с")
        print(f"🤖 Ответ: {response[:200]}...")
        
        print("\n✅ Тест нашего агента пройден!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании нашего агента: {e}")
        logger.error(f"Ошибка тестирования нашего агента: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов GigaChat Function Calling")
    print("Документация: https://developers.sber.ru/docs/ru/gigachat/guides/function-calling")
    
    # Тест базового function calling
    success1 = test_gigachat_function_calling()
    
    # Тест нашей реализации
    success2 = test_our_agent_implementation()
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"✅ Базовый function calling: {'ПРОЙДЕН' if success1 else 'ПРОВАЛЕН'}")
    print(f"✅ Наша реализация агента: {'ПРОЙДЕН' if success2 else 'ПРОВАЛЕН'}")
    
    if success1 and success2:
        print("\n🎉 Все тесты пройдены! Реализация соответствует документации Сбера.")
    else:
        print("\n⚠️ Некоторые тесты не пройдены. Проверьте реализацию.") 