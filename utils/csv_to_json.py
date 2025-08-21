import pandas as pd
import json
from .text_cleaning import clean_text

def convert_csv_to_json(csv_path: str, json_path: str):
    df = pd.read_csv(csv_path)
    processed = df[["title", "authors", "published_year", "average_rating", "description", "categories"]].copy()
    processed.rename(columns={
        "published_year": "publication_date",
        "average_rating": "rate",
        "description": "content"
    }, inplace=True)

    processed.dropna(subset=["title", "authors", "publication_date", "rate", "content", "categories"], inplace=True)
    for col in ["title", "authors", "publication_date", "rate", "content", "categories"]:
        processed = processed[processed[col].astype(str).str.strip() != ""]
    processed["content"] = processed["content"].apply(clean_text)

    processed.to_json(json_path, orient="records", indent=4, force_ascii=False)
    return df, processed
