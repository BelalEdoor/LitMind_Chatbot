import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def build_summarizer(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    return summarizer, tokenizer
