from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    @abstractmethod
    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str: ...

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs) -> Any:
        """Нативный ответ SDK — для доступа к специфичным фичам провайдера."""
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает chat_raw")

    @abstractmethod
    async def list_available_ids(self) -> list[str]: ...
