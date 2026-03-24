# run_baseline_manual_glue.py
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score

def main():
    model_name = "distilbert-base-uncased"
    seed = 42

    # Paths to local SST-2 TSVs
    train_path = "glue_data/SST-2/train.tsv"
    dev_path   = "glue_data/SST-2/dev.tsv"

    # Load with pandas
    train_df = pd.read_csv(train_path, sep="\t")
    val_df   = pd.read_csv(dev_path, sep="\t")

    # Wrap into HuggingFace DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"], truncation=True, padding="max_length", max_length=128
        )

    encoded = dataset.map(preprocess, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # TrainingArguments
    args = TrainingArguments(
        output_dir="./outputs/distilbert-sst2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        logging_dir="./logs",
        logging_steps=50,
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train & evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation:", metrics)

if __name__ == "__main__":
    main()
