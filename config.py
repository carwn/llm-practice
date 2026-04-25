from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Config:
    proxyapi_key: str
    proxyapi_base_url: str = "https://api.proxyapi.ru/openai/v1"
    ollama_host: str = "http://192.168.1.141:11434"
    catalog_path: str = "models/catalog.yaml"


def load_config() -> Config:
    key = os.getenv("PROXYAPI_KEY")
    if not key:
        raise ValueError("PROXYAPI_KEY не задан — добавьте в .env")
    return Config(
        proxyapi_key=key,
        proxyapi_base_url=os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://192.168.1.141:11434"),
        catalog_path=os.getenv("CATALOG_PATH", "models/catalog.yaml"),
    )
