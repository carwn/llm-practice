# LLM Practice

A Python CLI for accessing LLM models from multiple providers — **proxyapi.ru** (OpenAI, Anthropic, Google) and a **local Ollama** instance — through a single unified interface.

## Features

- Single command to chat with any model: GPT, Claude, Gemini, Llama, Qwen, DeepSeek
- Compare responses from different providers side-by-side
- Model catalog with tier, cost, and capabilities metadata
- Auto-discovery: sync catalog from Ollama and proxyapi.ru
- Works as CLI and as a Python library

## Setup

```bash
git clone https://github.com/carwn/llm-practice.git
cd llm-practice

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add PROXYAPI_KEY and OLLAMA_HOST
```

## Usage

```bash
# List available models
python cli.py models
python cli.py models --provider anthropic
python cli.py models --source ollama

# Chat
python cli.py chat "Explain quantum computing" --model gpt-4o-mini
python cli.py chat "Write a poem" --model claude-sonnet-4-6

# Compare same prompt across providers (parallel)
python cli.py compare "What is recursion?"
python cli.py compare "Write hello world in Rust" \
  --models "gpt-4o-mini,claude-sonnet-4-6,gemini-2.0-flash"
```

### Optional: shell alias

Add to `~/.zshrc` for shorter commands:
```bash
alias llm='"/path/to/llm-practice/.venv/bin/python3" "/path/to/llm-practice/cli.py"'
```

Then:
```bash
llm models
llm chat "Hello" --model qwen3:14b
llm compare "Explain gradient descent"
```

## Update Model Catalog

```bash
# Sync from proxyapi.ru (OpenAI + Anthropic + Google models)
python tools/discover_proxyapi.py

# Sync from local Ollama
python tools/discover_ollama.py
```

## Supported Sources

| Source               | Models                        | API         |
|----------------------|-------------------------------|-------------|
| `proxyapi-openai`    | GPT-4o, GPT-4.1, o3, o4…     | OpenAI SDK  |
| `proxyapi-anthropic` | Claude Sonnet, Opus, Haiku…   | Anthropic SDK |
| `proxyapi-google`    | Gemini 2.0, 2.5 Flash/Pro…   | Google genai SDK |
| `ollama`             | Llama, Qwen, DeepSeek, Mistral… | curl       |

## Library Usage

```python
from config import load_config
from client import LLMClient

client = LLMClient(load_config())
response = await client.chat_simple("gpt-4o-mini", "Hello!")
```

## Configuration

| Variable           | Description                          | Default                              |
|--------------------|--------------------------------------|--------------------------------------|
| `PROXYAPI_KEY`     | API key from proxyapi.ru             | required                             |
| `PROXYAPI_BASE_URL`| ProxyAPI OpenAI endpoint             | `https://api.proxyapi.ru/openai/v1`  |
| `OLLAMA_HOST`      | Local Ollama address                 | `http://192.168.1.100:11434`         |
| `CATALOG_PATH`     | Path to model catalog                | `models/catalog.yaml`                |
