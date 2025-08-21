import os
import gradio as gr
import json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from chatbot.chatbot_logic import chatbot, clear_chat
from utils.pdf_utils import pdf_to_json
from utils.csv_to_json import convert_csv_to_json

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/Data_en.csv"
JSON_PATH = "data/data_en.json"
SUMMARIES_PATH = "summaries_en.json"
MODEL_NAME = 'D:\\Trainings\\3- LLM & NLP Training (GSG)\\Module-13 (Final-Project)\\LitMinde_ChatBot\\fine_tuned_bart_optimized'

# -------------------------------
# Load your fine-tuned summarizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # use CPU; change to 0 for GPU if available
)

# -------------------------------
# Load CSV data
# -------------------------------
raw_df, json_df = convert_csv_to_json(DATA_PATH, JSON_PATH)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    books_data = json.load(f)

# -------------------------------
# PDF summarization helper
# -------------------------------
def summarize_pdf(pdf_file, summarizer):
    pdf_path = pdf_file.name
    temp_json = "temp_pdf.json"
    pdf_data = pdf_to_json(pdf_path, temp_json)
    summaries = []
    for page in pdf_data["pages"]:
        text = page["content"]
        summary = summarizer(
            text,
            max_length=150,
            min_length=50,
            do_sample=False,
            truncation=True
        )
        summary_text = summary[0]['summary_text']
        summaries.append(f"{summary_text}\n")
    return "\n".join(summaries)

# -------------------------------
# Build Gradio interface
# -------------------------------
def build_interface(books_data, summarizer, json_df, summaries_path):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 📚 LitMind")

        with gr.Row():
            chatbot_ui = gr.Chatbot(height=420, label="Conversation")
            state = gr.State([])

        with gr.Row():
            msg = gr.Textbox(placeholder="Type a message...", scale=5)
            upload_btn = gr.Button("📂 Choose PDF", variant="secondary", scale=2)
            pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"], visible=False, scale=2)
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)

        # Show PDF input when button clicked
        def toggle_upload():
            return gr.update(visible=False), gr.update(visible=True)

        upload_btn.click(fn=toggle_upload, inputs=None, outputs=[upload_btn, pdf_input])

        # Handle both chat messages and PDF summaries
        def unified_handler(message, chat_history, pdf_file):
            if pdf_file is not None:
                summary = summarize_pdf(pdf_file, summarizer)
                chat_history.append((f"📄 Summary of {os.path.basename(pdf_file.name)}", summary))
                return "", chat_history, None
            else:
                response, chat_history = chatbot(
                    message, chat_history, books_data, summarizer, json_df, summaries_path
                )
                return "", chat_history, pdf_file

        send.click(fn=unified_handler, inputs=[msg, state, pdf_input], outputs=[msg, chatbot_ui, pdf_input])
        msg.submit(fn=unified_handler, inputs=[msg, state, pdf_input], outputs=[msg, chatbot_ui, pdf_input])
        clear.click(fn=clear_chat, outputs=[chatbot_ui, state])

    return demo

# -------------------------------
# Launch
# -------------------------------
demo = build_interface(books_data, summarizer, json_df, SUMMARIES_PATH)
demo.launch()
