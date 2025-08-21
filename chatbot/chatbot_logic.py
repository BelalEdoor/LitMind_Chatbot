from typing import List
from utils.cache_utils import load_summaries, save_summary
from utils.text_cleaning import clean_text
import re

def summarize_book_with_prompt(user_prompt: str, books_data, summarizer, summaries_path: str):
    summaries = load_summaries(summaries_path)
    prompt_key = user_prompt.strip().lower()
    if prompt_key in summaries:
        return summaries[prompt_key]

    for book in books_data:
        if book.get("title", "").lower() in prompt_key:
            desc = clean_text(book.get("content") or "")
            summary = summarizer(f"{user_prompt}: {desc}", max_length=150, min_length=50, do_sample=False, truncation=True)
            summary_text = summary[0]['summary_text']
            final_summary = f"**Summary of {book['title']}**\n- Author: {book.get('authors','N/A')}\n- Published: {book.get('publication_date','N/A')}\n- Rating: {book.get('rate','N/A')}\n\n{summary_text}"
            save_summary(summaries_path, prompt_key, final_summary)
            return final_summary
    return f"Book or prompt '{user_prompt}' not found in database."

def chatbot(user_input: str, history: List[List[str]], books_data, summarizer, json_df, summaries_path):
    msg = user_input.strip()
    if msg.lower().startswith("summarize"):
        user_prompt = msg.replace("summarize", "").strip()
        response = summarize_book_with_prompt(user_prompt, books_data, summarizer, summaries_path)
    elif msg.lower().startswith("recommend"):
        from .recommend import recommend_books
        user_prompt = msg.replace("recommend", "").strip()
        response = recommend_books(user_prompt, json_df, summarizer, summaries_path)
    elif msg.lower() in ["hi", "hello"]:
        response = "Hello! You can ask me to 'summarize <book>' or 'recommend <category>', or upload a PDF."
    else:
        response = "Try: summarize The Hobbit, or recommend fantasy books."
    history.append([msg, response])
    return response, history

def clear_chat():
    return [], []
