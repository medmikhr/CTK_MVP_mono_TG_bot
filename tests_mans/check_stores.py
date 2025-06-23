#!/usr/bin/env python3
"""
Скрипт для проверки состояния векторных хранилищ
"""

import os
from dotenv import load_dotenv
from document_processor_langchain import PERSIST_DIR
from embeddings_manager import get_local_huggingface_embeddings
from langchain_chroma import Chroma

# Загрузка переменных окружения
load_dotenv()

def check_vector_stores():
    """Проверка состояния векторных хранилищ."""
    print("🔍 Проверка векторных хранилищ")
    print("=" * 40)
    
    try:
        # Инициализация embeddings
        print("Инициализация embeddings...")
        embeddings = get_local_huggingface_embeddings()
        print("✅ Embeddings инициализированы")
        
        # Проверка директории
        print(f"\nПроверка директории: {PERSIST_DIR}")
        if os.path.exists(PERSIST_DIR):
            print("✅ Директория существует")
            files = os.listdir(PERSIST_DIR)
            print(f"   Файлов в директории: {len(files)}")
            for file in files:
                print(f"   - {file}")
        else:
            print("❌ Директория не существует")
            return False
        
        # Инициализация хранилищ
        stores = {
            "dama": Chroma(collection_name="dama", persist_directory=PERSIST_DIR, embedding_function=embeddings),
            "ctk": Chroma(collection_name="ctk", persist_directory=PERSIST_DIR, embedding_function=embeddings),
            "sbf": Chroma(collection_name="sbf", persist_directory=PERSIST_DIR, embedding_function=embeddings)
        }
        
        print("\nПроверка хранилищ:")
        total_docs = 0
        
        for name, store in stores.items():
            try:
                count = store._collection.count()
                print(f"   {name.upper()}: {count} документов")
                total_docs += count
                
                # Тестовый поиск
                if count > 0:
                    docs = store.similarity_search("тест", k=1)
                    print(f"     ✅ Поиск работает")
                else:
                    print(f"     ⚠️  Хранилище пустое")
                    
            except Exception as e:
                print(f"   {name.upper()}: ❌ Ошибка - {e}")
        
        print(f"\nИтого документов: {total_docs}")
        
        if total_docs == 0:
            print("\n⚠️  Все хранилища пустые!")
            print("Рекомендации:")
            print("1. Запустите bulk_upload.py для загрузки документов")
            print("2. Или используйте agent_simple.py для работы без документов")
            return False
        else:
            print("\n✅ Хранилища готовы к работе")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка при проверке хранилищ: {e}")
        return False

if __name__ == "__main__":
    success = check_vector_stores()
    if not success:
        print("\n❌ Обнаружены проблемы с хранилищами")
    else:
        print("\n✅ Все проверки пройдены успешно") 