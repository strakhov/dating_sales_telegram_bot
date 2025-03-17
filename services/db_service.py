import pickle
from tortoise.transactions import in_transaction
from models import UserBuffer
from llama_index.legacy.memory import ChatMemoryBuffer
import logging

class BufferManager:
    @staticmethod
    async def get_user_buffer(user_id: str) -> UserBuffer:
        return await UserBuffer.get_or_create(user_id=user_id)

    @staticmethod
    async def serialize_buffer(buffer: ChatMemoryBuffer) -> bytes:
        return pickle.dumps(buffer)

    @staticmethod
    async def deserialize_buffer(data: bytes) -> ChatMemoryBuffer:
        if not data:
            return ChatMemoryBuffer.from_defaults(token_limit=4000)
        return pickle.loads(data)

    @staticmethod
    async def update_buffer(user_id: str, **kwargs):
        try:
            async with in_transaction():
                # Используем get_or_create, чтобы избежать ошибки, если запись отсутствует
                user, created = await UserBuffer.get_or_create(user_id=user_id)
                if "buffer" in kwargs:
                    kwargs["buffer_data"] = await BufferManager.serialize_buffer(kwargs.pop("buffer"))
                user.update_from_dict(kwargs)
                await user.save()
        except Exception as e:
            logging.error(f"Update buffer error: {str(e)}")
            raise