#!/usr/bin/env python3
"""
Примеры различных способов остановки программ с пользовательским вводом
"""

import signal
import sys
import time

def example_1_basic_input():
    """Базовый пример с input() - можно остановить только Ctrl+C"""
    print("=== Пример 1: Базовый input() ===")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        while True:
            user_input = input("Введите что-то: ")
            print(f"Вы ввели: {user_input}")
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем (Ctrl+C)")
    except EOFError:
        print("\nПрограмма остановлена (EOF)")

def example_2_with_exit_commands():
    """Пример с командами выхода"""
    print("\n=== Пример 2: С командами выхода ===")
    print("Введите 'exit', 'quit' или 'выход' для завершения")
    
    try:
        while True:
            user_input = input("Введите что-то: ")
            
            # Проверка команд выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                print("Выход по команде пользователя")
                break
            
            print(f"Вы ввели: {user_input}")
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем (Ctrl+C)")

def example_3_with_signal_handler():
    """Пример с обработчиком сигналов"""
    print("\n=== Пример 3: С обработчиком сигналов ===")
    
    def signal_handler(signum, frame):
        print(f"\nПолучен сигнал {signum}. Завершение программы...")
        sys.exit(0)
    
    # Регистрация обработчика сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            user_input = input("Введите что-то: ")
            print(f"Вы ввели: {user_input}")
    except EOFError:
        print("\nПрограмма остановлена (EOF)")

def example_4_with_timeout():
    """Пример с таймаутом (требует дополнительных библиотек)"""
    print("\n=== Пример 4: С таймаутом ===")
    print("Этот пример показывает концепцию таймаута")
    print("Для реализации нужны дополнительные библиотеки")
    
    try:
        while True:
            print("Введите что-то (или подождите 10 секунд): ", end='', flush=True)
            
            # В реальном приложении здесь был бы таймаут
            # Например, с использованием select или threading
            user_input = input()
            print(f"Вы ввели: {user_input}")
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем (Ctrl+C)")

def main():
    """Главная функция с меню выбора примеров"""
    print("Примеры остановки программ с пользовательским вводом")
    print("=" * 50)
    
    examples = [
        ("Базовый input()", example_1_basic_input),
        ("С командами выхода", example_2_with_exit_commands),
        ("С обработчиком сигналов", example_3_with_signal_handler),
        ("С таймаутом (концепция)", example_4_with_timeout),
    ]
    
    while True:
        print("\nВыберите пример:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Выход")
        
        try:
            choice = input("\nВаш выбор: ")
            
            if choice == '0':
                print("До свидания!")
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(examples):
                _, func = examples[choice_num - 1]
                func()
            else:
                print("Неверный выбор!")
                
        except ValueError:
            print("Введите число!")
        except KeyboardInterrupt:
            print("\n\nПрограмма завершена пользователем")
            break

if __name__ == "__main__":
    main() 