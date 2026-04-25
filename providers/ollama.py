import json
import subprocess
from .base import BaseProvider
from config import Config


class OllamaProvider(BaseProvider):
    def __init__(self, config: Config):
        self._host = config.ollama_host

    def _curl(self, path: str, body: dict | None = None) -> dict:
        url = f"{self._host}{path}"
        cmd = ["curl", "-s", "--connect-timeout", "10", url]
        if body is not None:
            cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"curl error: {result.stderr}")
        return json.loads(result.stdout)

    async def chat(self, model_id: str, messages: list[dict], **kwargs) -> str:
        data = self._curl("/v1/chat/completions", {
            "model": model_id,
            "messages": messages,
            **kwargs,
        })
        return data["choices"][0]["message"]["content"]

    async def chat_raw(self, model_id: str, messages: list[dict], **kwargs) -> dict:
        return self._curl("/v1/chat/completions", {
            "model": model_id,
            "messages": messages,
            **kwargs,
        })

    async def list_available_ids(self) -> list[str]:
        data = self._curl("/api/tags")
        return [m["name"] for m in data.get("models", [])]
