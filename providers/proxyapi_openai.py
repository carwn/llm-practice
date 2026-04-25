from openai import AsyncOpenAI
from .base import BaseProvider
from config import Config


class ProxyAPIOpenAIProvider(BaseProvider):
    def __init__(self, config: Config):
        self.client = AsyncOpenAI(
            api_key=config.proxyapi_key,
            base_url=config.proxyapi_base_url,
        )

    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs):
        return await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )

    async def list_available_ids(self) -> list[str]:
        models = await self.client.models.list()
        return [m.id for m in models.data]
