# LLM Practice — Project Guide

## What This Is

A Python CLI utility for accessing LLM models from two sources:
- **proxyapi.ru** — cloud proxy for OpenAI, Anthropic, and Google models
- **Ollama** — local network inference server

## Architecture

```
config.py                  — loads settings from .env
client.py                  — LLMClient: single entry point, routes by model source
cli.py                     — typer CLI: chat, models, compare commands
models/
  schema.py                — ModelInfo dataclass, ChatMetrics dataclass
  registry.py              — ModelRegistry: load/get/filter/upsert/save
  catalog.yaml             — model catalog (hand-curated)
providers/
  base.py                  — BaseProvider ABC: chat(), chat_raw(), chat_with_metrics(), list_available_ids()
  proxyapi_openai.py       — OpenAI SDK → api.proxyapi.ru/openai/v1
  proxyapi_anthropic.py    — Anthropic SDK → api.proxyapi.ru/anthropic/v1
  proxyapi_google.py       — Google genai SDK → api.proxyapi.ru/google/v1
  ollama.py                — curl subprocess → local Ollama /v1
```

## Routing Logic

`LLMClient` looks up `model.source` in `catalog.yaml` and dispatches to the matching provider:

| source              | provider class            | SDK used       |
|---------------------|---------------------------|----------------|
| `proxyapi-openai`   | ProxyAPIOpenAIProvider    | openai         |
| `proxyapi-anthropic`| ProxyAPIAnthropicProvider | anthropic      |
| `proxyapi-google`   | ProxyAPIGoogleProvider    | google-genai   |
| `ollama`            | OllamaProvider            | curl via subprocess |

To add a new SDK: create a new provider file implementing `BaseProvider`, register it in `client.py`.

## Environment

Copy `.env.example` to `.env` and fill in:
- `PROXYAPI_KEY` — from proxyapi.ru dashboard
- `OLLAMA_HOST` — IP of the machine running Ollama (default: `http://192.168.1.100:11434`)

```bash
cp .env.example .env
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional alias for convenience (add to `~/.zshrc`):
```bash
alias llm='"/path/to/LLM practice/.venv/bin/python3" "/path/to/LLM practice/cli.py"'
```

## CLI Commands

```bash
# List models from catalog
python cli.py models
python cli.py models --provider anthropic
python cli.py models --source ollama
python cli.py models --tier top

# Chat with a model
python cli.py chat "What is gradient descent?" --model gpt-4o-mini
python cli.py chat "Explain this" --model claude-sonnet-4-6 --system "You are a teacher"

# Compare same prompt across multiple models (runs in parallel)
python cli.py compare "Explain recursion in 2 sentences"
python cli.py compare "Write hello world" --models "gpt-4o,claude-sonnet-4-6,gemini-2.0-flash"
```

## ModelInfo Fields

| field           | description                                      |
|-----------------|--------------------------------------------------|
| `id`            | model identifier used in API calls               |
| `provider`      | who made it: openai, anthropic, google, meta...  |
| `source`        | where to call it: proxyapi-openai, ollama, etc.  |
| `tier`          | base / mid / top                                 |
| `context_window`| max tokens                                       |
| `cost_input`    | USD per 1M input tokens (0.0 for Ollama)         |
| `cost_output`   | USD per 1M output tokens                         |
| `capabilities`  | list: vision, tools, json_mode, reasoning        |
| `available`     | set to false to disable without removing         |

## Using as a Library

```python
import asyncio
from config import load_config
from client import LLMClient

client = LLMClient(load_config())

# Simple string-in, string-out
response = await client.chat_simple("gpt-4o-mini", "Hello!")

# Full messages format
response = await client.chat("claude-sonnet-4-6", [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"},
])

# Raw SDK response (for provider-specific features)
raw = await client.chat_raw("claude-sonnet-4-6", messages)

# Response with metrics: latency, tokens, cost
text, metrics = await client.chat_with_metrics("gpt-4o-mini", messages)
metrics.latency_ms    # float, milliseconds
metrics.input_tokens  # int | None
metrics.output_tokens # int | None
metrics.total_tokens  # int | None  (property: in + out)
metrics.cost_usd      # float | None  (calculated from catalog rates)
```
