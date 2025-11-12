import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import preprocess_emails
from src.constants import RANDOM_STATE, TEST_SIZE
from transformers import AutoTokenizer, Trainer, TrainingArguments, DistilBertForSequenceClassification
import torch
from torch.utils.data import Dataset
import evaluate
import numpy as np

def tokenize(tokenizer, text):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt' 
    )

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels),
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="binary"),
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="binary"),
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="binary")
    }

def bert_main():
    print("[INFO] BERT model loading and preprocessing data...")
    cols = ["subject", "body", "label"]
    df = pd.read_csv("data/Enron.csv", usecols=cols)
    # Use concat to stack datasets (merge performs joins and can drop rows)
    df = pd.concat(
        [
            df,
            pd.read_csv("data/Nazario.csv", usecols=cols),
            pd.read_csv("data/Nigerian_Fraud.csv", usecols=cols),
        ],
        ignore_index=True,
    )

    df.dropna(subset=['label'], inplace=True)

    x_train, x_other, y_train, y_other = preprocess_emails(df)

    x_train.drop(columns=["subject_len", "body_len", "exclaim_count"])
    x_other.drop(columns=["subject_len", "body_len", "exclaim_count"])

    x_val, x_test, y_val, y_test = train_test_split(
        x_other, y_other, test_size=0.5, random_state=RANDOM_STATE
    )

    print("[INFO] Tokenizing text data...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenize(tokenizer, x_train["text"].to_list())
    val_encodings = tokenize(tokenizer, x_val["text"].to_list())
    test_encodings = tokenize(tokenizer, x_test["text"].to_list())
    
    train_dataset = TextClassificationDataset(train_encodings, y_train.to_list())
    val_dataset = TextClassificationDataset(val_encodings, y_val.to_list())
    test_dataset = TextClassificationDataset(test_encodings, y_test.to_list())

    print("[INFO] Initializing BERT model...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, finetuning_task="text-classification", problem_type = "single_label_classification").to(device)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    print("[INFO] Evaluating BERT model...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Model Evaluation Summary (BERT):", test_results)
