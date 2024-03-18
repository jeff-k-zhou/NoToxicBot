import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

dataset = load_dataset("csv", data_files="train.csv")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
def preprocess(examples):
    return tokenizer(examples["comment_text"], truncation=True, max_length=512, padding="max_length")

tokenized_data = dataset.map(preprocess, batched=True)
tokenized_data = tokenized_data.remove_columns(["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
tokenized_data = tokenized_data.rename_column("toxic", "labels")
tokenized_data = tokenized_data["train"].train_test_split(test_size=0.2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


idlabel = {0: "not_toxic", 1: "toxic"}
labelid = {"not_toxic": 0, "toxic": 1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=idlabel, label2id=labelid)

training_args = TrainingArguments(
    output_dir="result_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()