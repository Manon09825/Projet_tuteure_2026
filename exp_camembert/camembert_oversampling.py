from src.data import load_data, augment_alt, map_labels, convert_labels_to_ids, make_dataset
from src.model import load_model
from src.preprocessing import tokenize2

import random
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset
from collections import Counter
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------

random.seed(42)
np.random.seed(42)


file_path = "data/all_texts.conll"
model_name = "camembert-base"

output_dir = "results_oversampling"


label2id, id2label = map_labels()

tokenizer = AutoTokenizer.from_pretrained(model_name)


print("Chargement des données...")

sentences, labels = load_data(file_path)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences,
    labels,
    test_size=0.20,
    random_state=42)

print('Distribution des étiquettes avant sur-représentation des ALT: ')
all_labels = [
    l
    for sent in train_labels
    for l in sent
    ]

counts = Counter(all_labels)
for k, v in counts.items():
    print(k, v)

print("Train avant sur-représentation des ALT: ", len(train_sentences))

train_sentences, train_labels = augment_alt(
    train_sentences,
    train_labels)

print("Train après sur-représentation des ALT: ", len(train_sentences))


all_labels = [l
    for sent in train_labels
    for l in sent
    ]

counts = Counter(all_labels)

print("Distribution des étiquettes après sur-représentation: ")

for k, v in counts.items():
    print(k, v)


train_ids = convert_labels_to_ids(train_labels, label2id)
test_ids = convert_labels_to_ids(test_labels, label2id)

train_dataset = Dataset.from_dict({
    "tokens": train_sentences,
    "tags": train_ids
    })

test_dataset = Dataset.from_dict({
    "tokens": test_sentences,
    "tags": test_ids
    })


train_dataset = train_dataset.map(
    tokenize2,
    batched=True)

test_dataset = test_dataset.map(
    tokenize2,
    batched=True)



# --- Calcul du poids de chaque classe par rapport aux étiquettes sur tout le corpus ---

all_sent_labels=[x for s in train_ids for x in s]

counts=Counter(all_sent_labels)

total=sum(counts.values())

weights=[]

labels_list = ['O',
        'B-ALTC', 'I-ALTC',
        'B-ALTE','I-ALTE',
        'B-NONALT', 'I-NONALT',
        'B-ALTS', 'I-ALTS']

for i in range(len(labels_list)):

    c=counts.get(i,1)

    weights.append(
        total/(len(labels_list)*c)
    )

weights=torch.tensor(
    weights,
    dtype=torch.float
)

print("Class weights:\n")

for i,w in enumerate(weights):

    print(
        id2label[i],
        float(w)
    )



model=AutoModelForTokenClassification.from_pretrained(

    model_name,
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id
)


class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels=inputs["labels"]

        outputs=model(**inputs)

        logits=outputs.logits

        loss_fct=nn.CrossEntropyLoss(

            weight=weights.to(model.device),
            ignore_index=-100
        )

        loss=loss_fct(

            logits.view(
                -1,
                model.config.num_labels
            ),

            labels.view(-1)
        )

        return (
            (loss,outputs)
            if return_outputs
            else loss
        )


args=TrainingArguments(

    output_dir=output_dir,

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_steps=50,

    report_to="none"
)


trainer=WeightedTrainer(

    model=model,

    args=args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=DataCollatorForTokenClassification(
        tokenizer
    )
)

print("Training...")

trainer.train()

# --- SAUVEGARDE DES MODÈLES ---

trainer.save_model("./models/camembert_oversampling")
tokenizer.save_pretrained("./models/camembert_oversampling")


pred=trainer.predict(test_dataset)

logits=pred.predictions

preds=np.argmax(
    logits,
    axis=-1
)

true=test_dataset["labels"]

true_flat=[]
pred_flat=[]
tokens_flat=[]

for sent_idx in range(len(true)):

    word_idx=0

    for tok_idx in range(len(true[sent_idx])):

        if true[sent_idx][tok_idx] != -100:

            true_flat.append(
                true[sent_idx][tok_idx]
            )

            pred_flat.append(
                preds[sent_idx][tok_idx]
            )

            if word_idx < len(test_sentences[sent_idx]):

                tokens_flat.append(
                    test_sentences[sent_idx][word_idx]
                )

                word_idx += 1


decoded_true=[
    id2label[x]
    for x in true_flat
]

decoded_pred=[
    id2label[x]
    for x in pred_flat
]

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


# --- Matrice de confusion etc ---
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
    "./outputs/cm_camembert_oversampling.png",
    dpi=300
)



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


txt_path = "./outputs/adverbiaux_predits_oversampling.txt"

with open(txt_path, "w", encoding="utf-8") as f:

    for sent_tokens, sent_true, sent_pred in zip(
        test_sentences,
        true,
        preds
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
