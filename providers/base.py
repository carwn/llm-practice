import time
from abc import ABC, abstractmethod
from typing import Any
from models.schema import ChatMetrics


class BaseProvider(ABC):
    @abstractmethod
    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str: ...

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs) -> Any:
        """Нативный ответ SDK — для доступа к специфичным фичам провайдера."""
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает chat_raw")

    async def chat_with_metrics(
        self, model_id: str, messages: list[dict], **kwargs
    ) -> tuple[str, ChatMetrics]:
        """Возвращает (текст, метрики). По умолчанию токены не извлекаются."""
        t0 = time.perf_counter()
        text = await self.chat(model_id, messages, **kwargs)
        return text, ChatMetrics(latency_ms=(time.perf_counter() - t0) * 1000)

    @abstractmethod
    async def list_available_ids(self) -> list[str]: ...
