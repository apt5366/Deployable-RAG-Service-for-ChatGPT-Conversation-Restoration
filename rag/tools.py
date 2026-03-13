import json
from pathlib import Path

DATA_PATH = Path("data/conversations.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    conversations = json.load(f)


def count_mentions(keyword: str):

    keyword = keyword.lower()

    count = 0

    for conv in conversations:
        text = json.dumps(conv).lower()

        if keyword in text:
            count += 1

    return count