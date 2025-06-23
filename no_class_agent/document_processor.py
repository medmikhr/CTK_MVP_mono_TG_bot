import os
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.schema import Document
import hashlib
from embeddings_manager import get_local_huggingface_embeddings

# Конфигурация
PERSIST_DIR = "chroma_db_huggingface"  # Директория для хранения базы данных Chroma
CHUNK_SIZE = 1000  # Размер чанка при разбиении текста
CHUNK_OVERLAP = 200  # Перекрытие между чанками

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение embeddings из общего менеджера
embeddings = get_local_huggingface_embeddings()

# Создаем словарь для хранения векторных хранилищ для разных коллекций
vectorstores = {}

def get_vectorstore(collection: str) -> Chroma:
    """Получает или создает векторное хранилище для указанной коллекции"""
    if collection not in vectorstores:
        vectorstores[collection] = Chroma(
            collection_name=collection,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return vectorstores[collection]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def filter_duplicates(docs: List[Document], collection: str) -> List[Document]:
    """Фильтрует документы, уже существующие в базе"""
    existing_hashes = set()
    vectorstore = get_vectorstore(collection)

    # Получаем хеши существующих документов
    if os.path.exists(PERSIST_DIR):
        existing_data = vectorstore.get()  # Получаем все данные из базы
        for metadata in existing_data["metadatas"]:
            if "doc_hash" in metadata:
                existing_hashes.add(metadata["doc_hash"])

    # Фильтрация новых документов
    unique_docs = []
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in existing_hashes:
            doc.metadata["doc_hash"] = content_hash  # Добавляем хеш в метаданные
            unique_docs.append(doc)

    return unique_docs

def load_document(file_path: str) -> List[Dict]:
    """Загрузка документа в зависимости от его типа."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_extension}")
            return []
        
        # Загрузка и разбиение документа
        documents = loader.load()
        splits = text_splitter.split_documents(documents)
        
        # Добавление метаданных
        for split in splits:
            split.metadata.update({
                'source': file_path,
                'file_type': file_extension[1:],
                'file_name': os.path.basename(file_path)
            })
        
        return splits
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа {file_path}: {str(e)}")
        return []

def process_document(file_path: str, collection: str) -> bool:
    """Обработка документа и сохранение в ChromaDB."""
    try:        
        # Загрузка и разбиение документа
        splits = load_document(file_path)
        if not splits:
            logger.error(f"Не удалось загрузить документ: {file_path}")
            return False
        
        # Фильтрация дубликатов
        unique_splits = filter_duplicates(splits, collection)
        if not unique_splits:
            logger.info(f"Все чанки из документа {file_path} уже существуют в коллекции {collection}")
            return True
        
        # Добавление уникальных документов в векторное хранилище
        vectorstore = get_vectorstore(collection)
        vectorstore.add_documents(unique_splits)
        
        logger.info(f"✅ Документ успешно обработан и добавлен в коллекцию {collection}: {os.path.basename(file_path)}")
        logger.info(f"   Добавлено {len(unique_splits)} новых чанков")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {str(e)}")
        return False

def search_documents(query: str, collection: str, n_results: int = 5) -> List[Dict]:
    """Поиск по документам."""
    try:
        # Получаем векторное хранилище для указанной коллекции
        vectorstore = get_vectorstore(collection)
        
        # Поиск в векторном хранилище
        results = vectorstore.similarity_search_with_score(
            query,
            k=n_results
        )
        
        # Форматирование результатов
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {e}")
        return []

def delete_document(document_id: str, collection: str) -> bool:
    """Удаление документа из базы данных."""
    try:
        vectorstore = get_vectorstore(collection)
        # Удаление документов по метаданным
        vectorstore.delete(
            filter={"source": document_id}
        )
        
        logger.info(f"Документ успешно удален: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {e}")
        return False

def get_document_info(collection: str = None) -> Dict:
    """Получение информации о сохраненных документах."""
    try:
        if collection:
            # Получаем информацию для конкретной коллекции
            vectorstore = get_vectorstore(collection)
            collection_data = vectorstore.get()
        else:
            # Получаем информацию по всем коллекциям
            all_documents = []
            total_documents = 0
            for coll in vectorstores.keys():
                vectorstore = get_vectorstore(coll)
                collection_data = vectorstore.get()
                if collection_data and collection_data['documents']:
                    sources = set()
                    for metadata in collection_data['metadatas']:
                        sources.add(metadata['source'])
                    total_documents += len(sources)
                    all_documents.extend([
                        {
                            "source": source,
                            "collection": coll,
                            "chunks": len([m for m in collection_data['metadatas'] if m['source'] == source])
                        }
                        for source in sources
                    ])
            return {
                "total_documents": total_documents,
                "documents": all_documents
            }

        if not collection_data or not collection_data['documents']:
            return {"total_documents": 0, "documents": []}
        
        # Подсчет уникальных документов по источнику
        sources = set()
        for metadata in collection_data['metadatas']:
            sources.add(metadata['source'])
        
        return {
            "total_documents": len(sources),
            "documents": [
                {
                    "source": source,
                    "chunks": len([m for m in collection_data['metadatas'] if m['source'] == source])
                }
                for source in sources
            ]
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о документах: {e}")
        return {"total_documents": 0, "documents": []}

def process_documents_from_folder(folder_path: str, collection: str, file_extensions: List[str] = None) -> Dict:
    """Обработка всех документов из указанной папки в коллекцию."""
    if file_extensions is None:
        file_extensions = ['.pdf', '.doc', '.docx', '.txt']
    
    if not os.path.exists(folder_path):
        logger.error(f"Папка не найдена: {folder_path}")
        return {"success": False, "error": "Папка не найдена", "processed": 0, "failed": 0}
    
    processed_files = []
    failed_files = []
    
    try:
        # Получаем список всех файлов в папке
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in file_extensions:
                    files.append(file_path)
        
        logger.info(f"Найдено {len(files)} файлов для обработки в папке {folder_path}")
        
        # Обрабатываем каждый файл
        for file_path in files:
            try:
                logger.info(f"Обрабатываю файл: {file_path}")
                success = process_document(file_path, collection)
                if success:
                    processed_files.append(file_path)
                    logger.info(f"✅ Файл успешно обработан: {file_path}")
                else:
                    failed_files.append(file_path)
                    logger.error(f"❌ Ошибка при обработке файла: {file_path}")
            except Exception as e:
                failed_files.append(file_path)
                logger.error(f"❌ Исключение при обработке файла {file_path}: {e}")
        
        result = {
            "success": True,
            "total_files": len(files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "processed_files": processed_files,
            "failed_files": failed_files,
            "collection": collection
        }
        
        logger.info(f"Обработка завершена. Обработано: {len(processed_files)}, Ошибок: {len(failed_files)}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документов из папки: {e}")
        return {"success": False, "error": str(e), "processed": 0, "failed": 0}

if __name__ == "__main__":
    # Пример обработки одного документа
    file_path = r"C:\Users\Administrator\Documents\СберДокументы\DAMA-DMBOK (2020).pdf"
    collection_name = "dama_dmbok"
    process_document(file_path, collection_name)
    
    # Пример обработки документов из папки
    folder_path = r"C:\Users\Administrator\Documents\СберДокументы\ctk_docs_raw"
    collection_name = "ctk_methodology"
    process_documents_from_folder(folder_path, collection_name) 