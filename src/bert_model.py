import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import clean_text
from src.constants import RANDOM_STATE, TEST_SIZE
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
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
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

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

    df["text"] = "Sender: "+df["sender"].apply(clean_text) + \
    " Receiver: " + df["receiver"].apply(clean_text) + \
    " Date: " + df["date"].apply(clean_text) + \
    " Subject: " + df["subject"].apply(clean_text) + \
    " Body: " + df["body"].apply(clean_text)
    
    x = df["text"]
    y = df["label"].astype("float")

    x_train, x_other, y_train, y_other = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_other, y_other, test_size=0.5, random_state=RANDOM_STATE
    )

    print("[INFO] Tokenizing text data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenize(tokenizer, x_train.tolist())
    val_encodings = tokenize(tokenizer, x_val.tolist())
    test_encodings = tokenize(tokenizer, x_test.tolist())
    
    train_dataset = TextClassificationDataset(train_encodings, y_train.tolist())
    val_dataset = TextClassificationDataset(val_encodings, y_val.tolist())
    test_dataset = TextClassificationDataset(test_encodings, y_test.tolist())

    print("[INFO] Initializing BERT model...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
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
