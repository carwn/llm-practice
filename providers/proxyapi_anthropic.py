import anthropic
from .base import BaseProvider
from config import Config

_ANTHROPIC_BASE_URL = "https://api.proxyapi.ru/anthropic"


class ProxyAPIAnthropicProvider(BaseProvider):
    def __init__(self, config: Config):
        self.client = anthropic.AsyncAnthropic(
            api_key=config.proxyapi_key,
            base_url=_ANTHROPIC_BASE_URL,
            # proxyapi принимает Authorization: Bearer вместо x-api-key
            default_headers={"Authorization": f"Bearer {config.proxyapi_key}"},
        )

    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str:
        system, filtered = _split_system(messages)
        response = await self.client.messages.create(
            model=model_id,
            messages=filtered,
            max_tokens=kwargs.pop("max_tokens", 4096),
            **({"system": system} if system else {}),
            **kwargs,
        )
        return response.content[0].text

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs):
        system, filtered = _split_system(messages)
        return await self.client.messages.create(
            model=model_id,
            messages=filtered,
            max_tokens=kwargs.pop("max_tokens", 4096),
            **({"system": system} if system else {}),
            **kwargs,
        )

    async def list_available_ids(self) -> list[str]:
        models = await self.client.models.list()
        return [m.id for m in models.data]


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = None
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            filtered.append(m)
    return system, filtered
