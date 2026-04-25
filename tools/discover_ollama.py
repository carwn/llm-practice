"""Синхронизирует catalog.yaml с моделями, доступными в Ollama."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models.registry import ModelRegistry
from models.schema import ModelInfo

PROVIDER_PREFIXES: dict[str, str] = {
    "qwen": "alibaba",
    "llama": "meta",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "gemma": "google",
    "phi": "microsoft",
    "codellama": "meta",
    "vicuna": "lmsys",
    "falcon": "tii",
}


def detect_provider(model_id: str) -> str:
    lower = model_id.lower()
    for prefix, provider in PROVIDER_PREFIXES.items():
        if lower.startswith(prefix):
            return provider
    return "unknown"


def fetch_models(host: str) -> list[dict]:
    result = subprocess.run(
        ["curl", "-s", "--connect-timeout", "5", f"{host}/api/tags"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"curl завершился с кодом {result.returncode}: {result.stderr}")
    return json.loads(result.stdout).get("models", [])


def discover():
    config = load_config()
    registry = ModelRegistry(config.catalog_path)

    print(f"Подключаюсь к Ollama: {config.ollama_host} ...")
    models_raw = fetch_models(config.ollama_host)
    print(f"Найдено моделей: {len(models_raw)}")

    added, updated = 0, 0
    for m in models_raw:
        model_id = m["name"]
        details = m.get("details", {})
        param_size = details.get("parameter_size", "")

        try:
            existing = registry.get(model_id)
            existing.available = True
            registry.upsert(existing, overwrite_manual_fields=False)
            updated += 1
        except KeyError:
            registry.upsert(ModelInfo(
                id=model_id,
                provider=detect_provider(model_id),
                source="ollama",
                display_name=f"{model_id} ({param_size})" if param_size else model_id,
                tier="base",
                context_window=8192,
                cost_input=0.0,
                cost_output=0.0,
                capabilities=[],
                available=True,
            ))
            added += 1
            print(f"  + {model_id} ({param_size})")

    registry.save()
    print(f"\nГотово. Добавлено: {added}, обновлено: {updated}.")


if __name__ == "__main__":
    discover()
