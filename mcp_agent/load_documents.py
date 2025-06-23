#!/usr/bin/env python3
"""
Скрипт для загрузки документов в векторные хранилища
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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

class DocumentLoader:
    """Класс для загрузки документов в векторные хранилища."""
    
    def __init__(self):
        """Инициализация загрузчика документов."""
        self.setup_embeddings()
        self.setup_vector_stores()
        self.setup_text_splitter()
        
    def setup_embeddings(self):
        """Настройка embeddings."""
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
        
    def setup_text_splitter(self):
        """Настройка разделителя текста."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("Разделитель текста настроен")
        
    def load_pdf_documents(self, directory: str, collection_name: str) -> List[Document]:
        """
        Загрузка PDF документов из директории.
        
        Args:
            directory: Путь к директории с PDF файлами
            collection_name: Название коллекции для метаданных
            
        Returns:
            Список документов
        """
        documents = []
        
        if not os.path.exists(directory):
            logger.warning(f"Директория {directory} не существует")
            return documents
        
        try:
            # Загружаем все PDF файлы из директории
            loader = DirectoryLoader(
                directory,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            
            loaded_docs = loader.load()
            logger.info(f"Загружено {len(loaded_docs)} документов из {directory}")
            
            # Добавляем метаданные
            for doc in loaded_docs:
                doc.metadata["collection"] = collection_name
                doc.metadata["file_type"] = "pdf"
                
            documents.extend(loaded_docs)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки документов из {directory}: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разделение документов на чанки.
        
        Args:
            documents: Список документов
            
        Returns:
            Список разделенных документов
        """
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Разделено {len(documents)} документов на {len(split_docs)} чанков")
            return split_docs
        except Exception as e:
            logger.error(f"Ошибка разделения документов: {e}")
            return documents
    
    def add_documents_to_store(self, documents: List[Document], store: Chroma, store_name: str):
        """
        Добавление документов в векторное хранилище.
        
        Args:
            documents: Список документов
            store: Векторное хранилище
            store_name: Название хранилища для логирования
        """
        try:
            if not documents:
                logger.warning(f"Нет документов для добавления в {store_name}")
                return
            
            # Очищаем существующие документы
            store._collection.delete(where={})
            logger.info(f"Очищено хранилище {store_name}")
            
            # Добавляем новые документы
            store.add_documents(documents)
            store.persist()
            
            logger.info(f"Добавлено {len(documents)} документов в {store_name}")
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов в {store_name}: {e}")
    
    def load_dama_documents(self, dama_directory: str):
        """
        Загрузка документов DAMA DMBOK.
        
        Args:
            dama_directory: Путь к директории с документами DAMA
        """
        print(f"\n📚 Загрузка документов DAMA DMBOK из {dama_directory}")
        print("=" * 50)
        
        # Загружаем документы
        documents = self.load_pdf_documents(dama_directory, "dama_dmbok")
        
        if not documents:
            print("⚠️  Документы DAMA не найдены")
            return
        
        # Разделяем на чанки
        split_docs = self.split_documents(documents)
        
        # Добавляем в хранилище
        self.add_documents_to_store(split_docs, self.dama_store, "DAMA DMBOK")
        
        print(f"✅ Загружено {len(split_docs)} чанков в хранилище DAMA DMBOK")
    
    def load_ctk_documents(self, ctk_directory: str):
        """
        Загрузка методологических материалов ЦТК.
        
        Args:
            ctk_directory: Путь к директории с материалами ЦТК
        """
        print(f"\n📚 Загрузка методологических материалов ЦТК из {ctk_directory}")
        print("=" * 50)
        
        # Загружаем документы
        documents = self.load_pdf_documents(ctk_directory, "ctk_methodology")
        
        if not documents:
            print("⚠️  Документы ЦТК не найдены")
            return
        
        # Разделяем на чанки
        split_docs = self.split_documents(documents)
        
        # Добавляем в хранилище
        self.add_documents_to_store(split_docs, self.ctk_store, "ЦТК")
        
        print(f"✅ Загружено {len(split_docs)} чанков в хранилище ЦТК")
    
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

def main():
    """Главная функция для загрузки документов."""
    print("🚀 Загрузчик документов для агента управления данными")
    print("=" * 60)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python load_documents.py <dama_directory> <ctk_directory>")
        print("")
        print("Пример:")
        print("  python load_documents.py ../dama_docs ../ctk_docs")
        print("")
        print("Если директории не указаны, будут использованы значения по умолчанию:")
        print("  DAMA: ../dama_docs")
        print("  ЦТК: ../ctk_docs")
        print("")
        
        # Используем значения по умолчанию
        dama_dir = "../dama_docs"
        ctk_dir = "../ctk_docs"
    else:
        dama_dir = sys.argv[1]
        ctk_dir = sys.argv[2]
    
    try:
        # Инициализируем загрузчик
        loader = DocumentLoader()
        
        # Проверяем текущее состояние хранилищ
        print("\n📊 Текущее состояние хранилищ:")
        store_info = loader.get_store_info()
        for store_name, info in store_info.items():
            if "error" not in info:
                status_icon = "✅" if info["status"] == "ready" else "⚠️"
                print(f"   {status_icon} {store_name}: {info['documents']} документов")
            else:
                print(f"   ❌ {store_name}: ошибка - {info['error']}")
        
        # Загружаем документы DAMA
        if os.path.exists(dama_dir):
            loader.load_dama_documents(dama_dir)
        else:
            print(f"\n⚠️  Директория DAMA не найдена: {dama_dir}")
        
        # Загружаем документы ЦТК
        if os.path.exists(ctk_dir):
            loader.load_ctk_documents(ctk_dir)
        else:
            print(f"\n⚠️  Директория ЦТК не найдена: {ctk_dir}")
        
        # Показываем финальное состояние
        print("\n📊 Финальное состояние хранилищ:")
        final_store_info = loader.get_store_info()
        for store_name, info in final_store_info.items():
            if "error" not in info:
                status_icon = "✅" if info["status"] == "ready" else "⚠️"
                print(f"   {status_icon} {store_name}: {info['documents']} документов")
            else:
                print(f"   ❌ {store_name}: ошибка - {info['error']}")
        
        print("\n✅ Загрузка завершена!")
        print("Теперь можно запустить агента: python agent.py")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 