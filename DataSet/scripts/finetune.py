"""
Fine-tuning client: upload train.jsonl → create fine-tuning job → poll status.

Usage:
    OPENAI_API_KEY=sk-... python DataSet/scripts/finetune.py
    OPENAI_API_KEY=sk-... python DataSet/scripts/finetune.py --poll <job-id>
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def upload_file(client, path: str) -> str:
    print(f"Uploading {path}...")
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    print(f"  File uploaded: {file_id}")
    return file_id


def create_job(client, file_id: str, model: str = "gpt-4o-mini-2024-07-18") -> str:
    print(f"Creating fine-tuning job (base model: {model})...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        hyperparameters={
            "n_epochs": 3,
        },
        suffix="ticket-classifier",
    )
    job_id = job.id
    print(f"  Job created: {job_id}")
    print(f"  Status: {job.status}")
    return job_id


def poll_job(client, job_id: str, interval: int = 30) -> None:
    print(f"Polling job {job_id} every {interval}s... (Ctrl+C to stop)")
    terminal_statuses = {"succeeded", "failed", "cancelled"}

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        trained_tokens = getattr(job, "trained_tokens", None)
        fine_tuned_model = getattr(job, "fine_tuned_model", None)

        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}", end="")
        if trained_tokens:
            print(f" | Trained tokens: {trained_tokens}", end="")
        print()

        if status == "succeeded":
            print(f"\nFine-tuned model: {fine_tuned_model}")
            save_result(job_id, fine_tuned_model, status)
            break
        elif status in terminal_statuses:
            print(f"\nJob ended with status: {status}")
            if job.error:
                print(f"Error: {job.error}")
            save_result(job_id, fine_tuned_model, status)
            break

        time.sleep(interval)


def save_result(job_id: str, model_id: str | None, status: str) -> None:
    out = {
        "job_id": job_id,
        "fine_tuned_model": model_id,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = Path("DataSet/results/finetune_job.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Result saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune gpt-4o-mini on ticket classifier dataset")
    parser.add_argument("--train", default="DataSet/data/train.jsonl", help="Training JSONL file")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune")
    parser.add_argument("--poll", metavar="JOB_ID", help="Poll existing job by ID (skip upload)")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    client = get_client()

    if args.poll:
        poll_job(client, args.poll, interval=args.interval)
        return

    train_path = args.train
    if not Path(train_path).exists():
        print(f"ERROR: training file not found: {train_path}")
        sys.exit(1)

    file_id = upload_file(client, train_path)
    job_id = create_job(client, file_id, model=args.model)
    poll_job(client, job_id, interval=args.interval)


if __name__ == "__main__":
    main()
