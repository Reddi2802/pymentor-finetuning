from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "beginner_python_raw.jsonl"
OUT_DIR = ROOT / "data" / "processed"
SEED = 42

def validate(row, line_number):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Line {line_number}: expected exactly 3 messages")
    expected = ["system", "user", "assistant"]
    roles = [message.get("role") for message in messages]
    if roles != expected:
        raise ValueError(f"Line {line_number}: roles must be {expected}")
    if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
        raise ValueError(f"Line {line_number}: every message needs non-empty text")

def main():
    rows = []
    for n, line in enumerate(RAW_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        validate(row, n)
        rows.append({"messages": row["messages"]})
    if len(rows) < 20:
        raise ValueError("Add at least 20 examples before splitting.")
    random.Random(SEED).shuffle(rows)
    n = len(rows)
    train_end, validation_end = int(n * 0.8), int(n * 0.9)
    splits = {"train": rows[:train_end], "validation": rows[train_end:validation_end], "test": rows[validation_end:]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in subset:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} examples to {path}")

if __name__ == "__main__":
    main()
