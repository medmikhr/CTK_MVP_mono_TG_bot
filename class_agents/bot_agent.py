import os
import logging
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from dotenv import load_dotenv
from document_processor import process_document, get_document_info, delete_document
from class_functions_agent import call_agent, get_agent_status

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# TELEGRAM_BOT_TOKEN=7654961332:AAGcz-4UuI2M8NYsTXj63CEbFDUryEXZA1I бот агент
# TELEGRAM_BOT_TOKEN=7046694193:AAH9uutjQmLBqpTs5JMLMWUvUQjI5HDUN-I old bot
# Получение токена из переменных окружения
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("Не найден токен бота в переменных окружения")

# Словарь коллекций и их пользователей
COLLECTIONS = {
    "dama_dmbok": [
        673473862,  # Замените на реальные ID пользователей DAMA DMBOK
    ],
    "ctk_methodology": [
        135727236,  # Замените на реальные ID пользователей ЦТК
    ]
}

# Словарь для хранения состояний пользователей
user_states = {}

def get_user_collection(user_id: int) -> str:
    """Определяет, к какой коллекции имеет доступ пользователь"""
    for collection, users in COLLECTIONS.items():
        if user_id in users:
            return collection
    return None

def get_main_keyboard(user_id: int = None):
    """Создает основную клавиатуру только для пользователей с правами"""
    # Проверяем, есть ли у пользователя доступ к коллекциям
    if not get_user_collection(user_id):
        return None  # Возвращаем None для пользователей без прав
    
    keyboard = [
        [KeyboardButton("/tools_list")],
        [KeyboardButton("/load_doc")],
        [KeyboardButton("/docs_list")],
        [KeyboardButton("/delete_doc")]
    ]
    
    # Добавляем кнопку загрузки только для пользователей с правами
    if get_user_collection(user_id):
        keyboard.insert(1, [KeyboardButton("/load_doc")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    """Обработчик команды /start"""
    
    # Проверяем права пользователя
    user_collection = get_user_collection(user_id)
    
    if user_collection:
        # Пользователь с правами
        collection_display_name = {
            "dama_dmbok": "DAMA DMBOK",
            "ctk_methodology": "ЦТК"
        }.get(user_collection, user_collection.upper())
        
        help_text = (
            "Привет! Я Data Governance бот для повышения культуры работы с данными.\n"
            "Использую GigaChat для улучшенной обработки и поиска информации.\n\n"
            f"У вас есть доступ к коллекции: {collection_display_name}"
        )
        await update.message.reply_text(help_text, reply_markup=get_main_keyboard(user_id))
    else:
        # Пользователь без прав - только текстовое общение
        help_text = (
            "Привет! Я Data Governance бот для повышения культуры работы с данными.\n"
            "Использую GigaChat для улучшенной обработки и поиска информации.\n\n"
            "Просто напишите ваш вопрос, и я постараюсь на него ответить!"
        )
        await update.message.reply_text(help_text)

async def tools_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /tools_list"""
    user_id = update.effective_user.id
    
    # Проверяем права пользователя
    if not get_user_collection(user_id):
        await update.message.reply_text(
            "⛔ У вас нет доступа к этой команде. Обратитесь к администратору для получения прав."
        )
        return
    
    # Получаем информацию о доступных функциях напрямую
    agent_status = get_agent_status()
    
    if agent_status["status"] == "ready":
        functions_info = agent_status["functions"]
        
        response = "🔧 Доступные инструменты:\n\n"
        
        for func in functions_info["functions"]:
            response += f"📌 **{func['name']}**\n"
            response += f"   {func['description']}\n\n"
        
        response += f"📊 Всего функций: {functions_info['total_functions']}"
        
        await update.message.reply_text(response)
    else:
        await update.message.reply_text(
            f"❌ Ошибка при получении информации об инструментах: {agent_status.get('error', 'Неизвестная ошибка')}"
        )

async def load_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /load_doc"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем, есть ли у пользователя доступ к какой-либо коллекции
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет прав для загрузки документов. "
            "Обратитесь к администратору для получения доступа."
        )
        return
    
    collection_display_name = {
        "dama_dmbok": "DAMA DMBOK",
        "ctk_methodology": "ЦТК"
    }.get(collection, collection.upper())
    
    user_states[user_id] = 'waiting_for_document'
    message_text = (
        f"📤 Пожалуйста, отправьте документ для загрузки в коллекцию {collection_display_name}.\n"
        "Поддерживаемые форматы: PDF, DOC, DOCX, TXT"
    )
    await update.message.reply_text(message_text)

async def docs_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /docs_list"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем права пользователя
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет доступа к этой команде. Обратитесь к администратору для получения прав."
        )
        return
    
    info = get_document_info(collection=collection)
    if info["total_documents"] == 0:
        message_text = f"📊 В коллекции {collection.upper()} пока нет документов."
    else:
        response = f"📊 Документы в коллекции {collection.upper()}: {info['total_documents']}\n\n"
        for doc in info["documents"]:
            response += f"📄 {os.path.basename(doc['source'])}\n"
            response += f"   Чанков: {doc['chunks']}\n"
        message_text = response
    await update.message.reply_text(message_text)

async def delete_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /delete_doc"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем права пользователя
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет прав для удаления документов. "
            "Обратитесь к администратору для получения доступа."
        )
        return
    
    # Получаем список документов
    info = get_document_info(collection=collection)
    if info["total_documents"] == 0:
        await update.message.reply_text(f"📊 В коллекции {collection.upper()} нет документов для удаления.")
        return
    
    # Формируем список документов с номерами
    response = f"🗑️ Выберите документ для удаления (напишите номер):\n\n"
    documents_list = []
    
    for i, doc in enumerate(info["documents"], 1):
        filename = os.path.basename(doc['source'])
        response += f"{i}. 📄 {filename} ({doc['chunks']} чанков)\n"
        documents_list.append(doc['source'])  # Сохраняем полный путь
    
    # Сохраняем список документов в контексте пользователя
    user_states[user_id] = {
        'state': 'waiting_for_delete_choice',
        'documents': documents_list,
        'collection': collection
    }
    
    await update.message.reply_text(response)

async def handle_delete_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора документа для удаления"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id not in user_states or user_states[user_id].get('state') != 'waiting_for_delete_choice':
        return False
    
    try:
        choice = int(text)
        user_data = user_states[user_id]
        documents = user_data['documents']
        collection = user_data['collection']
        
        if choice < 1 or choice > len(documents):
            await update.message.reply_text(f"❌ Неверный номер. Выберите от 1 до {len(documents)}")
            return True
        
        # Получаем document_id (полный путь к файлу)
        document_id = documents[choice - 1]
        filename = os.path.basename(document_id)
        
        # Удаляем документ
        if delete_document(document_id, collection):
            await update.message.reply_text(f"✅ Документ '{filename}' успешно удалён из коллекции {collection.upper()}")
        else:
            await update.message.reply_text(f"❌ Ошибка при удалении документа '{filename}'")
        
        # Сбрасываем состояние пользователя
        user_states.pop(user_id, None)
        return True
        
    except ValueError:
        await update.message.reply_text("❌ Пожалуйста, введите число")
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {e}")
        await update.message.reply_text("❌ Произошла ошибка при удалении документа")
        user_states.pop(user_id, None)
        return True

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик получения документов"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем, есть ли у пользователя доступ к какой-либо коллекции
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет прав для загрузки документов. "
            "Обратитесь к администратору для получения доступа."
        )
        return
    
    # Проверяем, ожидаем ли мы документ от этого пользователя
    if user_id not in user_states or user_states[user_id] != 'waiting_for_document':
        await update.message.reply_text(
            "❌ Пожалуйста, сначала нажмите кнопку '/load_doc' или используйте команду /load_doc"
        )
        return
    
    try:
        # Получаем информацию о файле
        file = await context.bot.get_file(update.message.document.file_id)
        file_name = update.message.document.file_name
        
        # Проверка расширения файла
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in allowed_extensions:
            await update.message.reply_text(
                f"❌ Неподдерживаемый формат файла: {file_ext}\n"
                f"Поддерживаемые форматы: {', '.join(allowed_extensions)}"
            )
            return
        
        # Скачиваем файл
        file_path = f"temp_{file_name}"
        await file.download_to_drive(file_path)
        
        # Обрабатываем документ
        collection_display_name = {
            "dama_dmbok": "DAMA DMBOK",
            "ctk_methodology": "ЦТК"
        }.get(collection, collection.upper())
        
        if process_document(file_path, collection=collection):
            await update.message.reply_text(f"✅ Документ успешно обработан и добавлен в коллекцию {collection_display_name}: {file_name}")
        else:
            await update.message.reply_text("❌ Ошибка при обработке документа")
        
        # Удаляем временный файл
        os.remove(file_path)
        
        # Сбрасываем состояние пользователя
        user_states.pop(user_id, None)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке документа")
        # Сбрасываем состояние пользователя в случае ошибки
        user_states.pop(user_id, None)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    try:
        user_id = update.effective_user.id
        text = update.message.text
        
        # Проверяем, не находится ли пользователь в состоянии выбора документа для удаления
        if await handle_delete_choice(update, context):
            return
        
        # Отправляем сообщение о том, что запрос обрабатывается
        processing_message = await update.message.reply_text("🤔 Обрабатываю ваш запрос с помощью GigaChat...")
        
        # Передаем запрос новому агенту
        agent_response = call_agent(text, str(user_id))
        
        # Удаляем сообщение о обработке
        await processing_message.delete()
        
        # Отправляем ответ
        if agent_response.success:
            # Добавляем информацию о времени обработки
            response_text = f"{agent_response.response}\n\n⏱️ Время обработки: {agent_response.processing_time:.2f}с"
            await update.message.reply_text(response_text)
        else:
            error_message = (
                f"❌ Произошла ошибка при обработке вашего запроса.\n"
                f"Ошибка: {agent_response.error}\n"
                f"Пожалуйста, попробуйте еще раз или обратитесь к администратору."
            )
            await update.message.reply_text(error_message)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте еще раз или обратитесь к администратору."
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    
    # Вызываем соответствующую функцию в зависимости от нажатой кнопки
    if query.data == "tools_list":
        await tools_list(update, context)
    elif query.data == "load_doc":
        await load_doc(update, context)
    elif query.data == "docs_list":
        await docs_list(update, context)
    elif query.data == "delete_doc":
        await delete_doc(update, context)
    
    # Отвечаем на callback query
    await query.answer()

async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик для всех остальных типов сообщений"""
    user_id = update.effective_user.id
    user_collection = get_user_collection(user_id)
    
    if user_collection:
        # Пользователь с правами
        await update.message.reply_text(
            "❌ Я могу обрабатывать только текстовые сообщения и документы.\n"
            "Пожалуйста, отправьте текстовый запрос или документ.\n\n"
            "Используйте кнопки меню для доступа к функциям."
        )
    else:
        # Пользователь без прав
        await update.message.reply_text(
            "❌ Я могу обрабатывать только текстовые сообщения.\n"
            "Пожалуйста, отправьте текстовый запрос."
        )

def main():
    """Основная функция запуска бота"""
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tools_list", tools_list))
    application.add_handler(CommandHandler("load_doc", load_doc))
    application.add_handler(CommandHandler("docs_list", docs_list))
    application.add_handler(CommandHandler("delete_doc", delete_doc))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.ALL, handle_other_messages))  # Обработчик для всех остальных типов сообщений
    application.add_handler(CallbackQueryHandler(button_callback))  # Обработчик кнопок
    
    # Запускаем бота
    logger.info("Запуск бота с GigaChat Functions Agent...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 