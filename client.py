from typing import Any
from config import Config
from models.registry import ModelRegistry
from providers.base import BaseProvider
from providers.proxyapi_openai import ProxyAPIOpenAIProvider
from providers.proxyapi_anthropic import ProxyAPIAnthropicProvider
from providers.proxyapi_google import ProxyAPIGoogleProvider
from providers.ollama import OllamaProvider


class LLMClient:
    def __init__(self, config: Config):
        self.registry = ModelRegistry(config.catalog_path)
        self._providers: dict[str, BaseProvider] = {
            "proxyapi-openai":    ProxyAPIOpenAIProvider(config),
            "proxyapi-anthropic": ProxyAPIAnthropicProvider(config),
            "proxyapi-google":    ProxyAPIGoogleProvider(config),
            "ollama":             OllamaProvider(config),
            # Добавить нативный провайдер: "anthropic-native": AnthropicProvider(config)
        }

    def _get_provider(self, model_id: str) -> BaseProvider:
        model = self.registry.get(model_id)
        if model.source not in self._providers:
            raise ValueError(
                f"Нет провайдера для source='{model.source}'. "
                f"Зарегистрированы: {list(self._providers)}"
            )
        return self._providers[model.source]

    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str:
        return await self._get_provider(model_id).chat(model_id, messages, **kwargs)

    async def chat_simple(self, model_id: str, prompt: str, **kwargs) -> str:
        return await self.chat(
            model_id,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs) -> Any:
        return await self._get_provider(model_id).chat_raw(model_id, messages, **kwargs)
