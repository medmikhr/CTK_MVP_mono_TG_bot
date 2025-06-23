import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_gigachat import GigaChatEmbeddings

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logger = logging.getLogger(__name__)

# Конфигурация
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"  # Модель для HuggingFace embeddings

# Глобальные переменные для хранения экземпляров embeddings
_huggingface_embeddings = None
_gigachat_embeddings = None
_local_huggingface_embeddings = None

def get_huggingface_embeddings():
    """Получение экземпляра HuggingFace embeddings (через endpoint, singleton)."""
    global _huggingface_embeddings
    
    if _huggingface_embeddings is None:
        HF_TOKEN = os.getenv('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError("Не найден токен HuggingFace в переменных окружения")
        
        logger.info("Инициализация HuggingFaceEndpointEmbeddings...")
        _huggingface_embeddings = HuggingFaceEndpointEmbeddings(
            repo_id=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_TOKEN
        )
        logger.info("HuggingFaceEndpointEmbeddings инициализированы")
    
    return _huggingface_embeddings

def get_local_huggingface_embeddings():
    """Получение экземпляра HuggingFaceEmbeddings (локально, singleton)."""
    global _local_huggingface_embeddings
    if _local_huggingface_embeddings is None:
        logger.info("Инициализация локальных HuggingFaceEmbeddings...")
        _local_huggingface_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Локальные HuggingFaceEmbeddings инициализированы")
    return _local_huggingface_embeddings

def get_gigachat_embeddings():
    """Получение экземпляра GigaChat embeddings (singleton)."""
    global _gigachat_embeddings
    
    if _gigachat_embeddings is None:
        GC_AUTH = os.getenv('GIGACHAT_TOKEN')
        if not GC_AUTH:
            raise ValueError("Не найден токен GigaChat в переменных окружения")
        
        logger.info("Инициализация GigaChat embeddings...")
        _gigachat_embeddings = GigaChatEmbeddings(
            credentials=GC_AUTH,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS"
        )
        logger.info("GigaChat embeddings инициализированы")
    
    return _gigachat_embeddings

def test_embeddings(embeddings_type="huggingface", local=False):
    """Тестирование работы embeddings."""
    try:
        if embeddings_type == "huggingface":
            if local:
                embeddings = get_local_huggingface_embeddings()
                test_text = "Тестовый текст для проверки работы локальных HuggingFace embeddings"
            else:
                embeddings = get_huggingface_embeddings()
                test_text = "Тестовый текст для проверки работы HuggingFace endpoint embeddings"
        elif embeddings_type == "gigachat":
            embeddings = get_gigachat_embeddings()
            test_text = "Тестовый текст для проверки работы GigaChat embeddings"
        else:
            raise ValueError(f"Неизвестный тип embeddings: {embeddings_type}")
        
        embedding = embeddings.embed_query(test_text)
        logger.info(f"Тест {embeddings_type} embeddings успешен. Размер вектора: {len(embedding)}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании {embeddings_type} embeddings: {e}")
        return False 