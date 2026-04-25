from pathlib import Path
from typing import Optional
import yaml
from .schema import ModelInfo


class ModelRegistry:
    def __init__(self, catalog_path: str):
        self._path = Path(catalog_path)
        self._models: dict[str, ModelInfo] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        for item in data:
            m = ModelInfo(**item)
            self._models[m.id] = m

    def get(self, model_id: str) -> ModelInfo:
        if model_id not in self._models:
            raise KeyError(f"Модель '{model_id}' не найдена в каталоге")
        return self._models[model_id]

    def filter(
        self,
        provider: Optional[str] = None,
        source: Optional[str] = None,
        tier: Optional[str] = None,
        capability: Optional[str] = None,
        available: Optional[bool] = None,
    ) -> list[ModelInfo]:
        result = list(self._models.values())
        if provider:
            result = [m for m in result if m.provider == provider]
        if source:
            result = [m for m in result if m.source == source]
        if tier:
            result = [m for m in result if m.tier == tier]
        if capability:
            result = [m for m in result if capability in m.capabilities]
        if available is not None:
            result = [m for m in result if m.available == available]
        return result

    def upsert(self, model: ModelInfo, overwrite_manual_fields: bool = False):
        if model.id in self._models and not overwrite_manual_fields:
            existing = self._models[model.id]
            # Не перезаписываем поля, выставленные вручную
            model.tier = existing.tier
            model.cost_input = existing.cost_input
            model.cost_output = existing.cost_output
            model.capabilities = existing.capabilities
            model.display_name = existing.display_name
        self._models[model.id] = model

    def save(self):
        data = []
        for m in sorted(self._models.values(), key=lambda x: (x.provider, x.id)):
            data.append({
                "id": m.id,
                "provider": m.provider,
                "source": m.source,
                "display_name": m.display_name,
                "tier": m.tier,
                "context_window": m.context_window,
                "cost_input": m.cost_input,
                "cost_output": m.cost_output,
                "capabilities": m.capabilities,
                "available": m.available,
            })
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    def all(self) -> list[ModelInfo]:
        return list(self._models.values())
