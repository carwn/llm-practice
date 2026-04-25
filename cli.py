#!/usr/bin/env python3
import asyncio
from typing import Optional
import typer
from config import load_config
from client import LLMClient

DEFAULT_COMPARE_MODELS = [
    "gpt-4o-mini",
    "claude-sonnet-4-6",
    "gemini-2.0-flash",
]

app = typer.Typer(add_completion=False, help="LLM CLI — доступ к моделям через ProxyAPI и Ollama")


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Запрос к модели"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="ID модели из каталога"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Отправить запрос к модели и получить ответ."""
    config = load_config()
    client = LLMClient(config)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = asyncio.run(client.chat(model, messages))
    typer.echo(response)


@app.command(name="models")
def list_models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Фильтр по провайдеру"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Фильтр по источнику"),
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Фильтр по классу: base/mid/top"),
):
    """Показать модели из каталога."""
    config = load_config()
    client = LLMClient(config)

    items = client.registry.filter(provider=provider, source=source, tier=tier, available=True)
    if not items:
        typer.echo("Модели не найдены")
        raise typer.Exit(1)

    typer.echo(f"{'ID':<45} {'TIER':<6} {'PROVIDER':<12} {'SOURCE':<18} {'$/1M in':<9} {'$/1M out'}")
    typer.echo("-" * 105)
    for m in sorted(items, key=lambda x: (x.provider, x.tier, x.id)):
        typer.echo(
            f"{m.id:<45} {m.tier:<6} {m.provider:<12} {m.source:<18} {m.cost_input:<9.3f} {m.cost_output:.3f}"
        )


@app.command(name="compare")
def compare_models(
    prompt: str = typer.Argument(..., help="Запрос ко всем моделям"),
    models: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Модели через запятую. По умолчанию: gpt-4o-mini,claude-sonnet-4-6,gemini-2.0-flash",
    ),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Отправить один запрос нескольким моделям и сравнить ответы."""
    model_ids = models.split(",") if models else DEFAULT_COMPARE_MODELS

    config = load_config()
    client = LLMClient(config)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async def run_all():
        tasks = [client.chat(mid.strip(), messages) for mid in model_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_all())

    width = 80
    typer.echo(f"\nПромпт: {prompt}\n")
    for model_id, result in zip(model_ids, results):
        typer.echo("─" * width)
        typer.echo(f"  {model_id}")
        typer.echo("─" * width)
        if isinstance(result, Exception):
            typer.echo(f"  ОШИБКА: {result}")
        else:
            for line in result.splitlines():
                typer.echo(f"  {line}")
        typer.echo()


if __name__ == "__main__":
    app()
