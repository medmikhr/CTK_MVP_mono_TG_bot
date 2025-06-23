#!/usr/bin/env python3
"""
Скрипт для массовой загрузки документов из папки в векторную базу данных.
Использует GigaChat embeddings для обработки документов.
"""

import os
import sys
import argparse
import logging
from typing import List
from document_processor_gigachat import (
    process_documents_from_folder,
    get_folder_info,
    test_embeddings,
    get_document_info
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_size(size_bytes: int) -> str:
    """Форматирование размера файла в читаемый вид."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def print_folder_info(folder_info: dict):
    """Вывод информации о папке."""
    if "error" in folder_info:
        print(f"❌ Ошибка: {folder_info['error']}")
        return
    
    print(f"\n📁 Папка: {folder_info['folder_path']}")
    print(f"📊 Всего файлов: {folder_info['total_files']}")
    print(f"💾 Общий размер: {format_size(folder_info['total_size'])}")
    print(f"📋 Поддерживаемые форматы: {', '.join(folder_info['supported_extensions'])}")
    
    if folder_info['files']:
        print("\n📄 Файлы:")
        for file_info in folder_info['files']:
            print(f"   • {file_info['name']} ({format_size(file_info['size'])})")

def print_upload_result(result: dict):
    """Вывод результата загрузки."""
    if not result.get("success", False):
        print(f"❌ Ошибка загрузки: {result.get('error', 'Неизвестная ошибка')}")
        return
    
    print(f"\n✅ Загрузка завершена!")
    print(f"📊 Статистика:")
    print(f"   • Всего файлов: {result['total_files']}")
    print(f"   • Успешно обработано: {result['processed']}")
    print(f"   • Ошибок: {result['failed']}")
    print(f"   • Коллекция: {result['collection']}")
    
    if result['processed_files']:
        print(f"\n✅ Успешно обработанные файлы:")
        for file_path in result['processed_files']:
            print(f"   • {os.path.basename(file_path)}")
    
    if result['failed_files']:
        print(f"\n❌ Файлы с ошибками:")
        for file_path in result['failed_files']:
            print(f"   • {os.path.basename(file_path)}")

def main():
    parser = argparse.ArgumentParser(
        description="Массовая загрузка документов из папки в векторную базу данных"
    )
    parser.add_argument(
        "folder_path",
        help="Путь к папке с документами"
    )
    parser.add_argument(
        "collection",
        help="Название коллекции для сохранения документов"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf", ".doc", ".docx", ".txt"],
        help="Поддерживаемые расширения файлов (по умолчанию: .pdf .doc .docx .txt)"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Только показать информацию о папке, без загрузки"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что будет загружено, без фактической загрузки"
    )
    
    args = parser.parse_args()
    
    # Проверяем существование папки
    if not os.path.exists(args.folder_path):
        print(f"❌ Папка не найдена: {args.folder_path}")
        sys.exit(1)
    
    # Получаем информацию о папке
    folder_info = get_folder_info(args.folder_path, args.extensions)
    print_folder_info(folder_info)
    
    if "error" in folder_info:
        sys.exit(1)
    
    if folder_info['total_files'] == 0:
        print(f"\n⚠️  В папке не найдено файлов с поддерживаемыми расширениями")
        print(f"Поддерживаемые форматы: {', '.join(args.extensions)}")
        sys.exit(0)
    
    if args.info_only:
        print(f"\nℹ️  Режим просмотра информации. Загрузка не выполнялась.")
        sys.exit(0)
    
    if args.dry_run:
        print(f"\n🔍 Режим предварительного просмотра:")
        print(f"Будет загружено {folder_info['total_files']} файлов в коллекцию '{args.collection}'")
        print(f"Для фактической загрузки запустите без флага --dry-run")
        sys.exit(0)
    
    # Проверяем подключение к GigaChat
    print(f"\n🔍 Проверка подключения к GigaChat...")
    if not test_embeddings():
        print("❌ Ошибка подключения к GigaChat. Проверьте токен и интернет-соединение.")
        sys.exit(1)
    print("✅ Подключение к GigaChat успешно")
    
    # Загружаем документы
    print(f"\n🚀 Начинаю загрузку документов в коллекцию '{args.collection}'...")
    result = process_documents_from_folder(args.folder_path, args.collection, args.extensions)
    
    # Выводим результат
    print_upload_result(result)
    
    # Показываем информацию о коллекции после загрузки
    if result.get("success", False) and result.get("processed", 0) > 0:
        print(f"\n📊 Информация о коллекции '{args.collection}' после загрузки:")
        collection_info = get_document_info(args.collection)
        print(f"   • Всего документов: {collection_info['total_documents']}")
        print(f"   • Модель embeddings: {collection_info.get('embedding_model', 'GigaChat')}")
        
        if collection_info['documents']:
            print(f"   • Документы:")
            for doc in collection_info['documents']:
                print(f"     - {os.path.basename(doc['source'])} ({doc['chunks']} чанков)")

if __name__ == "__main__":
    main() 