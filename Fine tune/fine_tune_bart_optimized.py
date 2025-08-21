import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# 0️⃣ Prepare data directory and CSVs
data_dir = "data"
train_path = os.path.join(data_dir, "train.csv")
val_path = os.path.join(data_dir, "val.csv")
full_csv = os.path.join(data_dir, "Data_en.csv")

# Use only part of the dataset to reduce training time
dataset_fraction = 0.5  # 50% of data
validation_split = 0.1  # 10% of this for validation

if not (os.path.exists(train_path) and os.path.exists(val_path)):
    print("train.csv or val.csv not found. Splitting Data_en.csv...")
    df = pd.read_csv(full_csv)
    df_small = df.sample(frac=dataset_fraction, random_state=42)
    train_df, val_df = train_test_split(df_small, test_size=validation_split, random_state=42)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"train.csv and val.csv created using {int(dataset_fraction*100)}% of the data!")

# 1️⃣ Load dataset
dataset = load_dataset("csv", data_files={"train": train_path, "validation": val_path})

# 2️⃣ Load tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 3️⃣ Preprocessing function with smaller max lengths
max_input_length = 64   # reduced from 128
max_target_length = 128 # reduced from 256

def preprocess_function(examples):
    inputs = [str(x) for x in examples["title"]]
    targets = [str(x) for x in examples["description"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )

    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 4️⃣ Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# 5️⃣ Training arguments with smaller batch size
training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,
    per_device_train_batch_size=4,  # reduced from 8
    per_device_eval_batch_size=4,   # reduced from 8
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7️⃣ Start training
trainer.train()

# 8️⃣ Save model & tokenizer
trainer.save_model("./fine_tuned_bart_optimized")
tokenizer.save_pretrained("./fine_tuned_bart_optimized")

print("Training finished and model saved to './fine_tuned_bart_optimized'")
