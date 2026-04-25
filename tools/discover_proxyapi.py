"""Синхронизирует catalog.yaml с моделями всех провайдеров на proxyapi.ru."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models.registry import ModelRegistry
from models.schema import ModelInfo
from providers.proxyapi_openai import ProxyAPIOpenAIProvider
from providers.proxyapi_anthropic import ProxyAPIAnthropicProvider
from providers.proxyapi_google import ProxyAPIGoogleProvider


async def discover():
    config = load_config()
    registry = ModelRegistry(config.catalog_path)

    sources = [
        ("proxyapi-openai",    "openai",    ProxyAPIOpenAIProvider(config)),
        ("proxyapi-anthropic", "anthropic", ProxyAPIAnthropicProvider(config)),
        ("proxyapi-google",    "google",    ProxyAPIGoogleProvider(config)),
    ]

    total_added, total_updated = 0, 0

    for source_key, default_provider, provider in sources:
        print(f"\n[{source_key}] Запрашиваю модели ...")
        try:
            ids = await provider.list_available_ids()
        except Exception as e:
            print(f"  Ошибка: {e}")
            continue

        print(f"  Найдено: {len(ids)}")
        added, updated = 0, 0
        for model_id in ids:
            try:
                existing = registry.get(model_id)
                existing.available = True
                registry.upsert(existing, overwrite_manual_fields=False)
                updated += 1
            except KeyError:
                registry.upsert(ModelInfo(
                    id=model_id,
                    provider=default_provider,
                    source=source_key,
                    display_name=model_id,
                    tier="mid",
                    context_window=128000,
                    cost_input=0.0,
                    cost_output=0.0,
                    capabilities=[],
                    available=True,
                ))
                added += 1
                print(f"    + {model_id}")
        total_added += added
        total_updated += updated
        print(f"  Добавлено: {added}, обновлено: {updated}")

    registry.save()
    print(f"\nГотово. Всего добавлено: {total_added}, обновлено: {total_updated}.")


if __name__ == "__main__":
    asyncio.run(discover())
