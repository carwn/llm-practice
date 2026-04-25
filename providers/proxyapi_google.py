from google import genai
from google.genai import types
from .base import BaseProvider
from config import Config

_GOOGLE_BASE_URL = "https://api.proxyapi.ru/google"


class ProxyAPIGoogleProvider(BaseProvider):
    def __init__(self, config: Config):
        self.client = genai.Client(
            api_key=config.proxyapi_key,
            http_options=types.HttpOptions(base_url=_GOOGLE_BASE_URL),
        )

    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str:
        response = await self.client.aio.models.generate_content(
            model=model_id,
            contents=_to_google_contents(messages),
            **kwargs,
        )
        return response.text

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs):
        return await self.client.aio.models.generate_content(
            model=model_id,
            contents=_to_google_contents(messages),
            **kwargs,
        )

    async def list_available_ids(self) -> list[str]:
        ids = []
        async for model in await self.client.aio.models.list():
            ids.append(model.name.removeprefix("models/"))
        return ids


def _to_google_contents(messages: list[dict]) -> list[types.Content]:
    role_map = {"user": "user", "assistant": "model", "system": "user"}
    return [
        types.Content(
            role=role_map.get(m["role"], "user"),
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
