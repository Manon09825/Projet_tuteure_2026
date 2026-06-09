import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    CamembertModel,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

from torchcrf import CRF

import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.data import load_data, map_span_labels, convert_labels_to_ids
from src.preprocessing import tokenize



random.seed(42)
np.random.seed(42)

file_path="data/first_texts_spans.conll"

model_name="camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
#tokenizer = AutoTokenizer.from_pretrained("./saved_camembert_crf")

labels_list=[
    "O",
    "B-SPAN",
    "I-SPAN"
]

label2id, id2label = map_span_labels()
sentences, labels = load_data(file_path)


# --- Séparation train test ----
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences,
    labels,
    test_size=0.30,
    random_state=42
)

print(
    "Train:",
    len(train_sentences)
)

print(
    "Test:",
    len(test_sentences)
)

# --- Conversion des étiquettes ---
train_ids=convert_labels_to_ids(
    train_labels,
    label2id
)

test_ids=convert_labels_to_ids(
    test_labels,
    label2id
)



# --- Création des datasets ---
train_dataset = Dataset.from_dict({
    "tokens": train_sentences,
    "tags": train_ids
})

train_dataset = train_dataset.map(tokenize, batched=True)

test_dataset=Dataset.from_dict({

    "tokens":test_sentences,
    "tags":test_ids
})

test_dataset=test_dataset.map(tokenize, batched=True)



# --- Mise en place du modèle ---
class CamembertCRF(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels
    ):

        super().__init__()

        self.camembert = CamembertModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.camembert.config.hidden_size
        )

        self.dropout = nn.Dropout(
            0.1
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

        self.crf = CRF(num_labels)



    def forward(self, input_ids=None, attention_mask=None, labels=None):

        outputs = self.camembert(

            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = (
            outputs.last_hidden_state
        )

        sequence_output = self.dropout(
            sequence_output
        )

        emissions = self.classifier(
            sequence_output
        )

        mask = attention_mask.to(torch.bool)

        # force sécurité CRF
        mask[:, 0] = True

        predictions = self.crf.decode(

            emissions,
            mask=mask
        )

        loss=None

        if labels is not None:

            labels_fixed=labels.clone()

            labels_fixed[
                labels_fixed==-100
            ]=0

            loss = -self.crf(

                emissions,
                labels_fixed,
                mask=mask,
                reduction='mean'
            )


        return {

            "loss":loss,
            "predictions":predictions
        }
        

model = CamembertCRF(model_name=model_name, num_labels=len(labels_list))

"""model=CamembertCRF(
    model_name="./saved_camembert_crf",
    num_labels=3
)"""



# --- Trainer ---
class CRFTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        outputs = model(**inputs)

        loss = outputs["loss"]

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


args = TrainingArguments(

    output_dir="./results_crf",

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_steps=20,

    report_to="none"
)


trainer = CRFTrainer(

    model=model,

    args=args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=DataCollatorForTokenClassification(
        tokenizer
    )
)

print("\nEntraînement...\n")

trainer.train()

# --- Sauvegarde du modèle pour ne pas devoir le ré-entraîner ---
print("Sauvegarde du modèle...")

trainer.save_model("./saved_camembert_crf")
tokenizer.save_pretrained("./saved_camembert_crf")

print("Modèle sauvegardé dans ./saved_camembert_crf")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

# --- Prédictions ---
preds = []

with torch.no_grad():
    for sample in test_dataset:

        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds.append(outputs["predictions"][0])


# --- Alignement true pred ---
true_flat = []
pred_flat = []

for i in range(len(test_dataset)):

    true_labels = test_dataset[i]["labels"]
    pred_labels = preds[i]

    j = 0

    for t in true_labels:

        if t == -100:
            continue

        true_flat.append(t)
        pred_flat.append(pred_labels[j])
        j += 1


# --- Classification report  et matrice de confusion---
decoded_true = [id2label[x] for x in true_flat]
decoded_pred = [id2label[x] for x in pred_flat]

print(classification_report(decoded_true, decoded_pred))

os.makedirs("./outputs", exist_ok=True)

cm = confusion_matrix(
    decoded_true,
    decoded_pred,
    labels=labels_list
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels_list,
    yticklabels=labels_list
)

plt.tight_layout()
plt.savefig("./outputs/confusion_matrix.png", dpi=300)



# --- Sauvegarde des spans prédits --
predicted_spans = []

for sent, pred in zip(test_sentences, preds):

    labels = [id2label[p] for p in pred]

    current = []

    for token, label in zip(sent, labels):

        if label == "B-SPAN":

            if current:
                predicted_spans.append(" ".join(current))

            current = [token]

        elif label == "I-SPAN":

            if current:
                current.append(token)

        else:
            if current:
                predicted_spans.append(" ".join(current))
                current = []

    if current:
        predicted_spans.append(" ".join(current))


os.makedirs("./outputs", exist_ok=True)

with open("./outputs/predicted_spans.txt", "w", encoding="utf-8") as f:
    for span in predicted_spans:
        f.write(span + "\n")
