import json
from utils.csv_to_json import convert_csv_to_json
from models.summarizer_model import build_summarizer
from interface.gradio_ui import build_interface
from pathlib import Path
from models.summarizer_model import build_summarizer

DATA_PATH = "data/Data_en.csv"
JSON_PATH = "data/data_en.json"
SUMMARIES_PATH = "summaries_en.json"


MODEL_NAME = 'D:\\Trainings\\3- LLM & NLP Training (GSG)\\Module-13 (Final-Project)\\LitMinde_ChatBot\\fine_tuned_bart_optimized'
summarizer, tokenizer = build_summarizer(str(MODEL_NAME))

# Load data
raw_df, json_df = convert_csv_to_json(DATA_PATH, JSON_PATH)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    books_data = json.load(f)

# Launch UI
demo = build_interface(books_data, summarizer, json_df, SUMMARIES_PATH)
demo.launch()
