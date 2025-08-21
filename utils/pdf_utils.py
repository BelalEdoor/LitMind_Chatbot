import json
import re
import pdfplumber
from .text_cleaning import clean_text

def pdf_to_json(pdf_path: str, json_path: str):
    data = {"pages": []}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            data["pages"].append({
                "page_number": i,
                "content": clean_text(text.strip())
            })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data
