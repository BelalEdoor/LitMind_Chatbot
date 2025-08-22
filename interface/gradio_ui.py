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
    # Custom Theme بدون text_color
    custom_theme = gr.themes.Base().set(
        body_background_fill="linear-gradient(to right, #e3f2fd, #bbdefb)",   # خلفية هادئة
        block_background_fill="white",                                       # صناديق بيضاء واضحة
        block_border_width="0px",
        block_shadow="0px 4px 15px rgba(0,0,0,0.08)",
        block_title_text_color="#1a237e",                                    # عنوان غامق أزرق
        button_primary_background_fill="linear-gradient(to right, #42a5f5, #1e88e5)",  # أزرار رئيسية أزرق متدرج
        button_primary_text_color="white",
        button_secondary_background_fill="linear-gradient(to right, #ffb74d, #fb8c00)", # أزرار ثانوية برتقالي متدرج
        button_secondary_text_color="white",
    )

    # Custom CSS للتحكم بألوان النصوص
    custom_css = """
    .gradio-container {max-width: 1100px; margin: auto; font-family: 'Segoe UI', sans-serif;}
    .gr-chatbot {background: #fafafa; border: 1px solid #e0e0e0; border-radius: 12px; padding: 10px;}
    .gr-button {font-weight: 600; border-radius: 8px; padding: 8px 12px;}
    h1, h2, h3, h4, h5, h6 {color: #1a237e;}
    p, label, .gr-textbox label, .gr-chatbot label {color: #212121;}
    """

    with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
        # Title
        gr.Markdown("""
        <div style="display: flex; align-items: center; gap: 12px;">
            <h1><b>LitMind Chatbot</b></h1>
        </div>
        <p>Welcome 👋<br>
        This chatbot helps you <b>summarize books and PDF files</b> quickly and smartly.</p>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(height=500, label="💬 Conversation")
                state = gr.State([])

                with gr.Row():
                    msg = gr.Textbox(placeholder="✍️ Type your message here...", scale=5)
                    send = gr.Button("➡️ Send", variant="primary", scale=1)
                    clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 📄 File Management")
                upload_btn = gr.Button("📂 Choose PDF", variant="secondary")
                pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"], visible=False)

        # Toggle upload
        def toggle_upload():
            return gr.update(visible=False), gr.update(visible=True)

        upload_btn.click(fn=toggle_upload, inputs=None, outputs=[upload_btn, pdf_input])

        # Unified handler
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

        # Footer
        gr.Markdown("<hr>")
        gr.Markdown("<p style='text-align:center;color:#616161'>🚀 Powered by <b>LitMind</b> | Made with ❤️ using HuggingFace & Gradio</p>")

    return demo



# -------------------------------
# Launch
# -------------------------------
demo = build_interface(books_data, summarizer, json_df, SUMMARIES_PATH)
demo.launch()
