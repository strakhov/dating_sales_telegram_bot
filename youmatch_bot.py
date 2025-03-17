from tortoise import Tortoise
from models import UserBuffer
from services.db_service import BufferManager
import config

import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import BotCommand, ContentType, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.filters.state import StateFilter

import openai
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from llama_index.legacy.memory import ChatMemoryBuffer
from llama_index.legacy import StorageContext
from llama_index.legacy import load_index_from_storage

# Import your keys for Azure OpenAI API
from keys import OPENAI_API_KEY, TG_TOKEN, CALENDLY_URL, CORPORATE_CHAT
from prompts import MAIN_PROMPT, STEP_WELCOME, STEP_RESULT, STEP_1, ABOUT_YOUMATCH, HELP_TEXT, COMMANDS_LIST

# Set up OpenAI environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.context_window = 4000

# Load index for general patient analysis
persist_dir = "data_index"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)

USER_LOCKS = {}
# Глобальный словарь для отслеживания ответа на followup
FOLLOWUP_RESPONDED = {}

def get_lock(user_id: str) -> asyncio.Lock:
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]

# Async function to get or initialize the user buffer
async def get_user_buffer(user_id):
    try:
        user, created = await UserBuffer.get_or_create(user_id=user_id)
        buffer = await BufferManager.deserialize_buffer(user.buffer_data)
        return {
            "buffer": buffer,
            "user_context": user.user_context,
            "lock": get_lock(user_id)  # добавили блокировку
        }
    
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        raise

# Async function to handle user message with RAG-agent
async def handle_user_message(user_id, message):
    # Ensure the user buffer is initialized before using the lock
    user_buffer = await get_user_buffer(user_id)
    
    async with user_buffer["lock"]:  # Now use the lock safely
        buffer = user_buffer["buffer"]  # Retrieve the buffer for the user
        prompt = MAIN_PROMPT

        # Create chat engine with updated syntax
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=buffer,
            system_prompt=prompt,
        )
        
        # Use the correct async method for generating a response
        response = await chat_engine.achat(message)  # Correct usage for async chat engine
        output = response.response  # Accessing the content of the response

        # saving context for clin_recs
        await BufferManager.update_buffer(
            user_id=user_id,
            user_context=output,
            buffer=buffer
        )
        return output


LOG_LEVEL = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARN, "error": logging.ERROR, "fatal": logging.FATAL}

def set_basic_logger():
    logging.basicConfig(
        level=LOG_LEVEL["info"],
        format="%(asctime)-20s : [proc %(process)s] : [%(levelname)-3s] : %(message)s",
    )

set_basic_logger()

# Определяем состояния анкеты
class QuestionnaireStates(StatesGroup):
    Q1 = State()  # Сколько вам лет?
    Q2 = State()  # Вы состоите сейчас в каких-либо отношениях? (Да/Нет)
    Q3 = State()  # Как давно завершили последние отношения?
    Q4 = State()  # Что, на ваш взгляд, мешает встретить свою вторую половинку?
    Q5 = State()  # Что вы изменили в себе за последние 2 года?
    Q6 = State()  # Насколько по 10-бальной шкале вы готовы к серьезным отношениям? Можете прокомментировать, почему.

class TelegramBot:
    def __init__(self):
        self.aiogram_bot = Bot(token=TG_TOKEN)
        self.dp = Dispatcher()  # No bot passed here
        self.register_handlers()

    async def init_db(self):
        await Tortoise.init(
            db_url=config.DATABASE_URL,
            modules={'models': ['models']}
        )
        await Tortoise.generate_schemas()

    # Метод для отложенной отправки followup-сообщений
    async def schedule_followup(self, chat_id: int, user_id: str):
        # Ждем 1 час
        await asyncio.sleep(3600)
        if not FOLLOWUP_RESPONDED.get(user_id, False):
            followup_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Да", callback_data="followup_yes"),
                 InlineKeyboardButton(text="Нет", callback_data="followup_no")]
            ])
            try:
                await self.aiogram_bot.send_message(
                    chat_id=chat_id,
                    text="Подскажите, удалось ли вам записаться на встречу?",
                    reply_markup=followup_keyboard
                )
            except Exception as e:
                logging.error(f"Ошибка при отправке первого followup-сообщения: {e}")
            # Ждем еще 6 часов
            await asyncio.sleep(6 * 3600)
            if not FOLLOWUP_RESPONDED.get(user_id, False):
                second_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="Да, хочу записаться на персональный разбор", url=CALENDLY_URL)]
                ])
                try:
                    await self.aiogram_bot.send_message(
                        chat_id=chat_id,
                        text=("К сожалению, я так и не получила от вас ответа. Бесплатная экспресс-диагностика "
                              "доступна только для ограниченного числа желающих. У нас осталось всего 7 мест.\n\n"
                              "Актуально это для вас сейчас?"),
                        reply_markup=second_keyboard
                    )
                except Exception as e:
                    logging.error(f"Ошибка при отправке второго followup-сообщения: {e}")

    def register_handlers(self):
        # 1. Обработчик команды /start – отправляем приветственное сообщение, видео и текст с кнопкой "Заполнить анкету"
        @self.dp.message(CommandStart())
        async def start_command(message: types.Message):
            user_id = str(message.from_user.id)
            logging.info(f"Received /start from user: {user_id}")

            # Сброс буфера для нового диалога
            await BufferManager.update_buffer(
                user_id=user_id,
                buffer=ChatMemoryBuffer.from_defaults(token_limit=4000),
                user_context=None
            )

            # Отправляем отдельное сообщение с текстом STEP_1 и кнопкой "Заполнить анкету"
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Записаться на интервью", url=CALENDLY_URL)]
                # [InlineKeyboardButton(text="Заполнить анкету", callback_data="start_questionnaire")]
            ])
            await message.answer(STEP_WELCOME, reply_markup=keyboard)

        # 2. Обработчик нажатия кнопки "Заполнить анкету" – начинаем опрос
        @self.dp.callback_query(lambda c: c.data == "start_questionnaire")
        async def start_questionnaire_callback(callback_query: types.CallbackQuery, state: FSMContext):
            await callback_query.message.answer("1. Сколько вам лет?")
            await state.set_state(QuestionnaireStates.Q1)
            await callback_query.answer()

        # 3. Обработка ответа на вопрос 1 (свободный текст: возраст)
        @self.dp.message(StateFilter(QuestionnaireStates.Q1))
        async def answer_q1(message: types.Message, state: FSMContext):
            await state.update_data(age=message.text)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Да", callback_data="q2_yes")],
                [InlineKeyboardButton(text="Нет", callback_data="q2_no")]
            ])
            await message.answer("2. Вы состоите сейчас в каких-либо отношениях? (Выберите вариант)", reply_markup=keyboard)
            await state.set_state(QuestionnaireStates.Q2)

        # 4. Обработка ответа на вопрос 2 – через callback_query
        @self.dp.callback_query(lambda c: c.data in ["q2_yes", "q2_no"], StateFilter(QuestionnaireStates.Q2))
        async def answer_q2_callback(callback_query: types.CallbackQuery, state: FSMContext):
            answer = "Да" if callback_query.data == "q2_yes" else "Нет"
            await state.update_data(relationship=answer)
            await callback_query.message.answer("3. Как давно завершили последние отношения?")
            await state.set_state(QuestionnaireStates.Q3)
            await callback_query.answer()

        # 5. Обработка ответа на вопрос 3 (свободный текст)
        @self.dp.message(StateFilter(QuestionnaireStates.Q3))
        async def answer_q3(message: types.Message, state: FSMContext):
            await state.update_data(last_relationship=message.text)
            await message.answer("4. Что, на ваш взгляд, мешает встретить свою вторую половинку?")
            await state.set_state(QuestionnaireStates.Q4)

        # 6. Обработка ответа на вопрос 4 (свободный текст)
        @self.dp.message(StateFilter(QuestionnaireStates.Q4))
        async def answer_q4(message: types.Message, state: FSMContext):
            await state.update_data(obstacles=message.text)
            await message.answer("5. Подскажите, а как вы сами изменились за последние 2 года?")
            await state.set_state(QuestionnaireStates.Q5)

        # 7. Обработка ответа на вопрос 5 (свободный текст)
        @self.dp.message(StateFilter(QuestionnaireStates.Q5))
        async def answer_q5(message: types.Message, state: FSMContext):
            await state.update_data(changes=message.text)
            await message.answer("6. Насколько по 10-бальной шкале вы готовы к серьезным отношениям? Можете прокомментировать, почему.")
            await state.set_state(QuestionnaireStates.Q6)

        # 8. Обработка ответа на вопрос 6 (свободный текст) и завершение анкеты
        @self.dp.message(StateFilter(QuestionnaireStates.Q6))
        async def answer_q6(message: types.Message, state: FSMContext):
            await state.update_data(ready=message.text)
            data = await state.get_data()
            user_id = str(message.from_user.id)
            summary = (
                f"Ответы пользователя {user_id}:\n"
                f"1. Сколько вам лет? {data.get('age')}\n"
                f"2. Вы состоите сейчас в отношениях? {data.get('relationship')}\n"
                f"3. Как давно завершили последние отношения? {data.get('last_relationship')}\n"
                f"4. Что мешает встретить вторую половинку? {data.get('obstacles')}\n"
                f"5. Что изменили в себе за последние 2 года? {data.get('changes')}\n"
                f"6. Готовность к серьезным отношениям (по 10-бальной шкале): {data.get('ready')}\n"
            )
            # Сохраняем summary в базу данных
            await BufferManager.update_buffer(user_id=user_id, user_context=summary)
            
            # Создаем клавиатуру с кнопкой "Записаться на консультацию"
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Записаться на персональный разбор", url=CALENDLY_URL)]
            ])
            
            # Отправляем сообщение с текстом STEP_RESULT и клавиатурой
            await message.answer(STEP_RESULT, reply_markup=keyboard)
            
            # Отправляем документ "content/info.pdf"
            try:
                pdf_file = FSInputFile("content/Гайд_«Проводник к осознанным и гармоничным отношениям».pdf")
                await message.answer_document(pdf_file)
            except Exception as e:
                logging.error(f"Ошибка при отправке документа: {e}")

            await state.clear()
            # Запускаем фоновую задачу для followup-сообщений
            asyncio.create_task(self.schedule_followup(message.chat.id, user_id))

        # 9. Обработчик нажатия кнопки "Хочу записаться на звонок" – переходим сразу на Calendly
        @self.dp.callback_query(lambda c: c.data == "book_call")
        async def book_call_callback(callback_query: types.CallbackQuery):
            message_text = (
                "Отлично! Для записи выберите удобное время через наш сервис: "
                f"{CALENDLY_URL}\n\n"
                "Ждем вас на звонке!"
            )
            await callback_query.message.answer(message_text)
            await callback_query.answer()

        # 10. Команда /bookcall – переводит пользователя по ссылке на Calendly
        @self.dp.message(Command("bookcall"))
        async def bookcall_command(message: types.Message):
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Записаться на звонок", url=CALENDLY_URL)]
            ])
            await message.answer("Пожалуйста, выберите удобное время для звонка:", reply_markup=keyboard)

        # 11. Команда /help – Связаться с экспертом
        @self.dp.message(Command("help"))
        async def help_command(message: types.Message):
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Связаться с экспертом", url="https://t.me/youmatch_concierge")]
            ])
            await message.answer(HELP_TEXT, reply_markup=keyboard)

        # 12. Команда /info – Подробнее о YouMatch
        @self.dp.message(Command("info"))
        async def info_command(message: types.Message):
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Интервью основателя YouMatch", url="https://youtu.be/q58np51o4Pk?feature=shared")]
            ])
            await message.answer(ABOUT_YOUMATCH, reply_markup=keyboard)

        # 13. Команда /profile – Заполнить анкету
        @self.dp.message(Command("profile"))
        async def profile_command(message: types.Message):
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Заполнить анкету", callback_data="start_questionnaire")]
            ])
            await message.answer(STEP_1, reply_markup=keyboard)

        # 14. Команда /menu – Список команд
        @self.dp.message(Command("menu"))
        async def menu_command(message: types.Message):

            await message.answer(COMMANDS_LIST)

        # 15. Обработчик followup-сообщений
        @self.dp.callback_query(lambda c: c.data == "followup_yes")
        async def followup_yes_handler(callback_query: types.CallbackQuery):
            user_id = str(callback_query.from_user.id)
            FOLLOWUP_RESPONDED[user_id] = True
            await callback_query.message.answer(
                "Отлично! На вашу почту была отправлена ссылка, по которой будет проходить встреча 🙂 Поздравляем вас с первым шагом на пути к здоровым отношениям!"
            )
            await callback_query.answer()

        @self.dp.callback_query(lambda c: c.data == "followup_no")
        async def followup_no_handler(callback_query: types.CallbackQuery):
            user_id = str(callback_query.from_user.id)
            FOLLOWUP_RESPONDED[user_id] = True
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Записаться на персональный разбор", url=CALENDLY_URL)]
            ])
            await callback_query.message.answer(
                "Обратите внимание, бесплатный разбор доступен для ограниченного числа желающих. У нас осталось всего 10 мест.\nЕсли у вас технические сложности нажмите /help для связи с вашим персональным ассистентом.",
                reply_markup=keyboard
            )
            await callback_query.answer()

        # 16. Общий обработчик текстовых сообщений – срабатывает только если пользователь не находится в режиме анкеты
        @self.dp.message(F.text)
        async def process_new_message(message: types.Message, state: FSMContext):
            current_state = await state.get_state()
            if current_state is not None:
                # Если пользователь находится в режиме анкеты, не обрабатываем это сообщение здесь
                return
            user_id = str(message.from_user.id)
            logging.info(f"Received text message from user {user_id}: {message.text}")
            try:
                # Обработка сообщения через RAG-агент (функция handle_user_message)
                response = await handle_user_message(user_id, message.text)
                await message.reply(response)
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
                await message.reply("Произошла ошибка обработки запроса")

    async def main(self):
        try:
            await self.aiogram_bot.set_my_commands([
                BotCommand(command="start", description="Начать работу"),
                BotCommand(command="bookcall", description="Записаться на звонок"),
                BotCommand(command="profile", description="Заполнить анкету"),
                BotCommand(command="help", description="Связаться с экспертом"),
                BotCommand(command="info", description="Подробнее о YouMatch"),
                BotCommand(command="menu", description="Список команд")
            ])
            await self.dp.start_polling(self.aiogram_bot)
        except Exception as e:
            logging.error(e)

async def main():
    bot = TelegramBot()
    await bot.init_db()
    try:
        await bot.main()
    finally:
        await Tortoise.close_connections()

if __name__ == "__main__":
    asyncio.run(main())