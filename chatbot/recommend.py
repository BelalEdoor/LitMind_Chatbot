import pandas as pd
import re
from datetime import datetime
from utils.text_cleaning import clean_text
from utils.cache_utils import load_summaries, save_summary

def score_row(row: pd.Series) -> float:
    score = 0.0
    if pd.notna(row.get("rate")):
        val = float(row["rate"])
        score += min(1.0, val / 5.0)
    if pd.notna(row.get("publication_date")):
        try:
            years_ago = max(0.0, (datetime.utcnow().year - int(row["publication_date"])))
            recency = 1.0 / (1.0 + years_ago)
            score += 0.2 * recency
        except Exception:
            pass
    return float(score)

def recommend_books(user_prompt: str, json_df, summarizer, summaries_path, k: int = 5):
    # استخراج الفئة من الـ prompt باستخدام regex
    match = re.search(r'([a-zA-Z]+)', user_prompt)
    category = match.group(1).lower() if match else None
    if not category:
        return "Sorry, I couldn't detect the category in your request."

    subset = json_df[json_df["categories"].str.contains(category, case=False, na=False)]
    if subset.empty:
        return f"No books found in category '{category}'."
    subset = subset.copy()
    subset["__score__"] = subset.apply(score_row, axis=1)
    top = subset.sort_values("__score__", ascending=False).head(k)

    recs = []
    for _, row in top.iterrows():
        desc = clean_text(row["content"] if pd.notna(row["content"]) else "No description")
        summaries = load_summaries(summaries_path)
        title_lower = row["title"].lower()
        if title_lower in summaries:
            summary_text = summaries[title_lower]
        else:
            summary = summarizer(f"{user_prompt}: {desc}", max_length=120, min_length=40, do_sample=False, truncation=True)
            summary_text = summary[0]['summary_text']
            final_summary = f"**Summary of {row['title']}**\n- Author: {row['authors']}\n- Published: {row['publication_date']}\n- Rating: {row['rate']}\n\n{summary_text}"
            save_summary(summaries_path, row["title"], final_summary)
        recs.append(f"📘 **{row['title']}** — {row['authors']} (⭐ {row['rate']})\nSummary: {summary_text}")

    return "\n\n".join(recs)
