import json
from pathlib import Path

INPUT = Path("token/sanskrit_tokens.json")
OUTPUT = Path("token/vocab.json")

# Special tokens
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<+>": 2,  # Sandhi marker, already part of your list
}

# Read original tokens
with open(INPUT, encoding="utf-8") as f:
    raw_tokens = json.load(f)

# Add all unique tokens with IDs starting from 3
next_id = max(vocab.values()) + 1
for entry in raw_tokens:
    token = entry["token"]
    if token not in vocab:
        vocab[token] = next_id
        next_id += 1

# Save vocab.json
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"✅ Saved vocab with {len(vocab)} tokens → {OUTPUT}")
