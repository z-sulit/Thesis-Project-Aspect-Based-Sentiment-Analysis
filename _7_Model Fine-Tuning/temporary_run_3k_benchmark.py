import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification; from torch.optim import AdamW
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
df_train = pd.read_csv('outputs/train_split.csv')
df_test = pd.read_csv('Test_Set_15.csv')

# Clean ground truth
def star_to_label(r):
    r = int(r)
    if r <= 2: return 'NEGATIVE'
    if r == 3: return 'NEUTRAL'
    return 'POSITIVE'
df_test['ground_truth'] = df_test['rating'].apply(star_to_label)

label2id = {'POSITIVE': 0, 'NEUTRAL': 1, 'NEGATIVE': 2}
id2label = {v: k for k, v in label2id.items()}

# 2. Train SVM
print("Training SVM...")
pre = ColumnTransformer([
    ('sentence', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True), 'sentence'),
    ('aspect',   OneHotEncoder(handle_unknown='ignore'), ['TargetAspect']),
])
svm = Pipeline([('pre', pre), ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=20000))])
svm.fit(df_train[['sentence', 'TargetAspect']], df_train['label'])
svm_preds_idx = svm.predict(df_test[['sentence', 'TargetAspect']])
df_test['pred_svm'] = [id2label[i] for i in svm_preds_idx]

# 3. Train mBERT
print("Training mBERT (this takes ~30 seconds)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            df['TargetAspect'].tolist(),
            df['sentence'].tolist(),
            truncation=True, padding=True, max_length=128, return_tensors='pt'
        )
        self.labels = torch.tensor(df['label'].tolist()) if 'label' in df else None
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

train_dataset = ABSADataset(df_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(4):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

print("Predicting with mBERT...")
test_dataset = ABSADataset(df_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
mbert_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        mbert_preds.extend(preds)
df_test['pred_mbert'] = [id2label[i] for i in mbert_preds]

# 4. VADER
print("Running VADER...")
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
def vader_predict(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05: return 'POSITIVE'
    elif score <= -0.05: return 'NEGATIVE'
    else: return 'NEUTRAL'
df_test['pred_vader'] = df_test['sentence'].apply(vader_predict)

# 5. XLM-RoBERTa
print("Loading XLM-RoBERTa predictions...")
xlmr_df = pd.read_csv('outputs/external_test_predictions.csv')
df_test['pred_xlmr'] = xlmr_df['pred_label']

# 6. Calculate Metrics
print("\n=== FINAL 3K DATASET BENCHMARK ===")
y_true = df_test['ground_truth']
labels_list = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']

results = []
for model_name, col in [('XLM-RoBERTa', 'pred_xlmr'), ('mBERT', 'pred_mbert'), ('SVM + TF-IDF', 'pred_svm'), ('VADER', 'pred_vader')]:
    y_pred = df_test[col]
    acc = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, labels=labels_list, average='macro', zero_division=0)
    w_f1 = f1_score(y_true, y_pred, labels=labels_list, average='weighted', zero_division=0)
    per_f1 = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    
    results.append({
        'Model': model_name,
        'accuracy': f'{acc:.3f}',
        'macro_f1': f'{mac_f1:.3f}',
        'weighted_f1': f'{w_f1:.3f}',
        'f1_positive': f'{per_f1[0]:.3f}',
        'f1_neutral': f'{per_f1[1]:.3f}',
        'f1_negative': f'{per_f1[2]:.3f}'
    })

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))
res_df.to_csv('outputs/3k_dataset_benchmark.csv', index=False)
print("\nSaved to: outputs/3k_dataset_benchmark.csv")
