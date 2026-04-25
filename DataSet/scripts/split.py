"""Split raw.jsonl into train (80%) and eval (20%) with stratification."""
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    raw_path = Path("DataSet/data/raw.jsonl")
    train_path = Path("DataSet/data/train.jsonl")
    eval_path = Path("DataSet/data/eval.jsonl")

    with open(raw_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # Group by label for stratified split
    by_label: dict[str, list] = defaultdict(list)
    for ex in examples:
        label = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        by_label[label].append(ex)

    train, eval_ = [], []
    random.seed(42)
    for label, items in by_label.items():
        random.shuffle(items)
        n_eval = max(1, round(len(items) * 0.2))
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    random.shuffle(train)
    random.shuffle(eval_)

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter

    train_labels = Counter(
        next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        for ex in train
    )
    eval_labels = Counter(
        next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        for ex in eval_
    )

    print(f"Total: {len(examples)} examples")
    print(f"Train: {len(train)} ({len(train)/len(examples)*100:.0f}%) — {dict(train_labels)}")
    print(f"Eval:  {len(eval_)} ({len(eval_)/len(examples)*100:.0f}%) — {dict(eval_labels)}")
    print(f"Saved: {train_path}, {eval_path}")


if __name__ == "__main__":
    main()
