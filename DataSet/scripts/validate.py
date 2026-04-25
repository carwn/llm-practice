"""Validate JSONL dataset files: JSON syntax, roles, content, labels."""
import json
import sys
from pathlib import Path
from collections import Counter

VALID_LABELS = {"bug", "feature", "billing", "question", "other"}
REQUIRED_ROLES = {"system", "user", "assistant"}
MIN_CONTENT_LEN = 5
MIN_ASSISTANT_LEN = 2  # labels can be as short as "bug" (3 chars)
MAX_CONTENT_LEN = 4096


def validate_file(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}")
        return False

    errors = []
    labels = []
    seen = set()

    with open(p) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                errors.append(f"Line {lineno}: empty line")
                continue

            # JSON validity
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"Line {lineno}: invalid JSON — {e}")
                continue

            # Top-level structure
            if "messages" not in obj:
                errors.append(f"Line {lineno}: missing 'messages' key")
                continue

            msgs = obj["messages"]
            if not isinstance(msgs, list) or len(msgs) < 3:
                errors.append(f"Line {lineno}: 'messages' must be list with ≥3 items")
                continue

            # Roles
            roles = {m.get("role") for m in msgs}
            missing = REQUIRED_ROLES - roles
            if missing:
                errors.append(f"Line {lineno}: missing roles: {missing}")
                continue

            # Content checks
            bad_content = False
            for m in msgs:
                role = m.get("role", "?")
                content = m.get("content", "")
                min_len = MIN_ASSISTANT_LEN if role == "assistant" else MIN_CONTENT_LEN
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"Line {lineno}: empty content in role='{role}'")
                    bad_content = True
                elif len(content) < min_len:
                    errors.append(f"Line {lineno}: content too short in role='{role}': {len(content)} chars")
                    bad_content = True
                elif len(content) > MAX_CONTENT_LEN:
                    errors.append(f"Line {lineno}: content too long in role='{role}': {len(content)} chars")
                    bad_content = True
            if bad_content:
                continue

            # Assistant label
            assistant_content = next(
                (m["content"].strip() for m in msgs if m.get("role") == "assistant"), None
            )
            if assistant_content not in VALID_LABELS:
                errors.append(
                    f"Line {lineno}: invalid label '{assistant_content}' (valid: {VALID_LABELS})"
                )
                continue

            # Duplicate detection
            user_content = next(
                (m["content"] for m in msgs if m.get("role") == "user"), ""
            )
            key = (user_content[:300], assistant_content)
            if key in seen:
                errors.append(f"Line {lineno}: duplicate example detected")
                continue
            seen.add(key)

            labels.append(assistant_content)

    total = lineno if "lineno" in dir() else 0
    valid = len(labels)

    print(f"\nFile: {path}")
    print(f"  Total lines : {total}")
    print(f"  Valid       : {valid}")
    print(f"  Errors      : {len(errors)}")
    if labels:
        print(f"  Distribution: {dict(Counter(labels))}")

    if errors:
        print("\nErrors found:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False

    print("  Status: OK")
    return True


def main():
    files = sys.argv[1:] or [
        "DataSet/data/raw.jsonl",
        "DataSet/data/train.jsonl",
        "DataSet/data/eval.jsonl",
    ]
    all_ok = True
    for f in files:
        if Path(f).exists():
            ok = validate_file(f)
            all_ok = all_ok and ok
        else:
            print(f"Skipping (not found): {f}")

    print("\n" + ("ALL VALID" if all_ok else "VALIDATION FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
