from dataclasses import dataclass, field


@dataclass
class ChatMetrics:
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


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
