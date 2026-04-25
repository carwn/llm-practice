from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    id: str
    provider: str           # "openai" | "anthropic" | "google" | "meta" | ...
    source: str             # "proxyapi" | "ollama" | "anthropic-native" | "google-native"
    display_name: str
    tier: str               # "base" | "mid" | "top"
    context_window: int
    cost_input: float       # USD за 1M input-токенов (0.0 для ollama)
    cost_output: float      # USD за 1M output-токенов
    capabilities: list[str] = field(default_factory=list)  # ["vision", "tools", "json_mode", "reasoning"]
    available: bool = True
