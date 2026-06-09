import numpy as np
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# --- CHARGEMENT DES DONNÉES ---

def load_conll(path):
    sentences = []
    sentence = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            parts = line.split()
            token = parts[0]
            label = parts[-1]
            sentence.append((token, label))

    if sentence:
        sentences.append(sentence)

    return sentences


# --- EXTRACTION DES SPANS GRÂCE AUX LABELS (SANS LES LABELS BIO) ----

def extract_spans(sentences):
    spans = []

    for sent in sentences:
        tokens, labels = zip(*sent)

        current_tokens = []
        current_label = labels[0]

        for tok, lab in zip(tokens, labels):

            if lab == current_label:
                current_tokens.append(tok)
            else:
                span_text = " ".join(current_tokens).strip()
                spans.append((span_text, current_label))

                current_tokens = [tok]
                current_label = lab

        span_text = " ".join(current_tokens).strip()
        spans.append((span_text, current_label))

    return spans

# --- PONDÉRATION (AJOUTÉE APRÈS LE PREMIER TEST) ----

def balance_data(spans, max_o_ratio=0.5):
    grouped = defaultdict(list)

    for text, label in spans:
        grouped[label].append(text)

    # downsample O
    o_samples = grouped["O"]
    other_samples = []

    for k, v in grouped.items():
        if k != "O":
            other_samples.extend([(x, k) for x in v])

    max_o = int(len(other_samples) * (max_o_ratio / (1 - max_o_ratio)))
    o_samples = random.sample(o_samples, min(len(o_samples), max_o))

    balanced = [(x, "O") for x in o_samples] + other_samples
    random.shuffle(balanced)

    return balanced


# --- MODÈLE CAMEMBERT ---

model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()


# --- PIPELINE ---

sentences = load_conll("textes/all_texts.conll")

spans = extract_spans(sentences)

spans = balance_data(spans)

texts = [x[0] for x in spans]
labels = [x[1] for x in spans]

X = np.array([embed(t) for t in tqdm(texts)])

# ----- TRAIN TEST ---

indices = np.arange(len(texts))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    labels,
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# sklearn Logistic Regression handles imbalance indirectly
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

# ---- ÉVALUATION ---

y_pred = clf.predict(X_test)

print("\n Classification report ")
print(classification_report(y_test, y_pred))

print("\n Confusion matrix ")

os.makedirs("./outputs", exist_ok=True)

cm_labels = sorted(list(set(y_test)))

cm = confusion_matrix(y_test, y_pred, labels=cm_labels)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=cm_labels,
    yticklabels=cm_labels
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("./outputs/confusion_matrix.png", dpi=300)

plt.show()

print("\n Métriques globales ")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")



indices = random.sample(range(len(X_test)), 10)

print("\n Quelques exemples de prédictions \n")

indices_sample = random.sample(range(len(X_test)), 30)

for i in indices_sample:
    original_idx = idx_test[i]

    print(f"Span       : {texts[original_idx]}")
    print(f"Vrai label : {y_test[i]}")
    print(f"Prédit     : {y_pred[i]}")
    print("-" * 40)
