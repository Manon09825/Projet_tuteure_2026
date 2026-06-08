from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random

from src.data import load_data, convert_labels_to_ids, map_labels
from src.preprocessing import tokenize


random.seed(42)
np.random.seed(42)

#  --------- CHARGEMENT DES DONNÉES ---------

file_path = "data/all_texts.conll"
model_name = "camembert-base"

output_dir = "results_finetuning_camemBERT"


label2id, id2label = map_labels()


print("Chargement des données...")

sentences, labels = load_data(file_path)

label2id, id2label = map_labels()

label_ids = convert_labels_to_ids(
    labels,
    label2id
)


dataset = Dataset.from_dict({
    "tokens": sentences,
    "tags": label_ids
})

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)


# ------ TOKENISATION ----------
"""tokenizer = AutoTokenizer.from_pretrained("camembert-base")"""

tokenizer = AutoTokenizer.from_pretrained(
    "./models/camembert_sans_ponderation"
)

tokenized_dataset = dataset.map(
    lambda batch: tokenize(batch, tokenizer),
    batched=True
)

word_ids_dataset = tokenized_dataset["test"]["word_ids"]

# ------ CHARGEMENT DU MODÈLE -------

labels_list = ['O',
        'B-ALTC', 'I-ALTC',
        'B-ALTE','I-ALTE',
        'B-NONALT', 'I-NONALT',
        'B-ALTS', 'I-ALTS']

""" LE MODÈLE EST DÉJÀ CHARGÉ DANS ./models/camembert_sans_ponderation

model = AutoModelForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id,
)"""

model = AutoModelForTokenClassification.from_pretrained(
    "./models/camembert_sans_ponderation"
)


# ----- ENTRAÎNEMENT ------

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

#trainer.train()


# ----- SAUVEGARDE DES POIDS DU MODÈLE -----

trainer.save_model("./models/camembert_sans_ponderation")
tokenizer.save_pretrained("./models/camembert_sans_ponderation")


# ---- PRÉDICTION -----

predictions = trainer.predict(tokenized_dataset["test"])

logits = predictions.predictions
pred_ids = np.argmax(logits, axis=-1)

true_ids = tokenized_dataset["test"]["labels"]

decoded_true = []
decoded_pred = []

for sent_true, sent_pred in zip(true_ids, pred_ids):

    for true_id, pred_id in zip(sent_true, sent_pred):

        # ignore tokens spéciaux
        if true_id == -100:
            continue

        decoded_true.append(id2label[true_id])
        decoded_pred.append(id2label[pred_id])



# ---- ÉVALUATION ET VISUALISATION ----


print("Classification report: ")

print(

classification_report(
    decoded_true,
    decoded_pred
)
)

precision,recall,f1,_ = precision_recall_fscore_support(

    decoded_true,
    decoded_pred,
    average="weighted"
)

print("Accuracy:",accuracy_score(decoded_true,decoded_pred))
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)



os.makedirs(
    "./outputs",
    exist_ok=True
)

cm=confusion_matrix(

    decoded_true,
    decoded_pred,
    labels=labels_list
)

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,
    annot=True,
    fmt='d',
    xticklabels=labels_list,
    yticklabels=labels_list
)

plt.tight_layout()

plt.savefig(
    "./outputs/camembert_sans_ponderation.png",
    dpi=300
)


logits = predictions.predictions
pred_ids = np.argmax(logits, axis=-1)

true_labels = tokenized_dataset["test"]["labels"]
test_tokens = tokenized_dataset["test"]["tokens"]


# Extraction des adverbiaux prédits pour les mettre dans un fichier txt

def extract_predicted_spans(tokens, pred_labels):

    spans = []

    current_tokens = []
    current_type = None

    for token, label in zip(tokens, pred_labels):

        if label == "O":

            if current_tokens:
                spans.append(
                    (
                        " ".join(current_tokens),
                        current_type
                    )
                )

            current_tokens = []
            current_type = None
            continue

        if label.startswith("B-"):

            if current_tokens:
                spans.append(
                    (
                        " ".join(current_tokens),
                        current_type
                    )
                )

            current_type = label[2:]
            current_tokens = [token]

        elif label.startswith("I-"):

            entity_type = label[2:]

            if (
                current_tokens
                and current_type == entity_type
            ):
                current_tokens.append(token)

            else:
                current_type = entity_type
                current_tokens = [token]

    if current_tokens:
        spans.append(
            (
                " ".join(current_tokens),
                current_type
            )
        )

    return spans


txt_path = "./outputs/adverbiaux_predits.txt"

with open(txt_path, "w", encoding="utf-8") as f:

    for sent_tokens, sent_true, sent_pred in zip(
        test_tokens,
        true_labels,
        pred_ids
    ):

        pred_labels_sentence = []

        token_idx = 0

        for true_id, pred_id in zip(sent_true, sent_pred):

            if true_id == -100:
                continue

            pred_labels_sentence.append(
                id2label[pred_id]
            )

            token_idx += 1

        spans = extract_predicted_spans(
            sent_tokens,
            pred_labels_sentence
        )

        for text, label in spans:

            f.write(
                f"{text}\t{label}\n"
            )



# Récupération des prédictions pour les sauvegarder dans des fichiers



print("pred shape: ", predictions.predictions.shape)
print(pred_ids.shape)

print(len(test_tokens))
print(len(true_labels))
print(len(pred_ids))

print(pred_ids[0].shape)


tsv_path = "./outputs/predictions.tsv"

with open(tsv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")

    writer.writerow(["token", "true", "pred"])

    for sent_tokens, sent_true, sent_pred in zip(
        test_tokens,
        true_labels,
        pred_ids
    ):

        print("tokens :", len(sent_tokens))
        print("labels :", len(sent_true))
        print("preds  :", len(sent_pred))


        token_idx = 0

        for true_id, pred_id in zip(sent_true, sent_pred):

            # ignore les tokens spéciaux CamemBERT
            if true_id == -100:
                continue

            token = sent_tokens[token_idx]

            writer.writerow([
                token,
                id2label[true_id],
                id2label[pred_id]
            ])

            token_idx += 1

        writer.writerow([])





















































