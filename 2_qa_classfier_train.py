import json
import csv
import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

files = [
    ("benchmark_data/nq_rear.json", "nq_rear", 0),
    ("benchmark_data/popqa.json", "popqa", 0),
    ("benchmark_data/2wikimultihopqa.json", "2wiki", 1),
    ("benchmark_data/hotpotqa.json", "hotpotqa", 1),
]

rows = []
for file, source, label in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "question" in item:
            rows.append({"question": item["question"], "source": source, "label": label})

random.shuffle(rows)

with open("train_questions_labeled.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["question", "source", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"已保存 {len(rows)} 条问题到 train_questions_labeled.csv")

# BERT微调部分
csv_path = "train_questions_labeled.csv"
df = pd.read_csv(csv_path)

# 划分训练/测试集
train_df, test_df = train_test_split(df, test_size=0.95, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["question"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = QuestionDataset(train_df)
test_dataset = QuestionDataset(test_df)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
training_args = TrainingArguments(
    output_dir='train_cat_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",  # 修正参数名
    logging_dir='train_cat_logs',
    logging_steps=50,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 保存训练后的模型
trainer.save_model("./trained_bert_all5_question_classifier")

# 测试集评估
metrics = trainer.evaluate()
print("测试集评估结果：", metrics)
