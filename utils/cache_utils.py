import json
import os
import re

def load_summaries(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_summary(path, key, summary):
    safe_key = re.sub(r'[\\/*?:"<>|]', "_", key)  # يجعل الاسم صالحًا للملفات
    data = load_summaries(path)
    data[safe_key] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
