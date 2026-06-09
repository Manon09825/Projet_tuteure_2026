import sklearn_crfsuite
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Get sentences and labels
conll_path = Path('textes/all_texts_bio.conll')

X = []
y = []

sent = []
sent_labels = []

with open(conll_path, 'r', encoding='utf-8') as file:
    for line in file:
        l = line.strip()
        parts = l.split()
        try:
            token = parts[0]
            sent.append(token)
#
            if parts[-1] in ['-', '_']:
                label = "O"
                sent_labels.append(label)
                print(f"Adding {label}")

            else:
                label = parts[-1]
                sent_labels.append(label)
                print(f"Adding {label}")


            if '.' in token:                
                X.append(sent)
                y.append(sent_labels)

                sent = []
                sent_labels = []
        
        except:
            print("No token or label found at line:", line)


# Filtrage par phrase
X_filtered = []
y_filtered = []

for tokens, labels in zip(X, y):
    has_alt = any(lab != "O" for lab in labels)

    if has_alt:
        for i in range(2):
            X_filtered.append(tokens)
            y_filtered.append(labels)

    else:
        continue

X = X_filtered
y = y_filtered


# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# Entraînement et prédiction
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True,
)

crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)


# Évaluation
pred_labels = []
true_labels = []

for i in range(len(y_pred)):
    for j in range(len(y_test[i])):
        true_labels.append(y_test[i][j])
        pred_labels.append(y_pred[i][j])


labels = []
for lab in pred_labels:
    if lab not in labels:
        labels.append(lab)
print(classification_report(true_labels, pred_labels, labels=labels ))


cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=labels,
    yticklabels=labels,
    cmap='Blues',
)

plt.title('Matrice de confusion modèle CRF')
plt.ylabel('True')
plt.xlabel('Predicted')

output_path = 'cm_base_crf.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Matrice de confusion enregistrée sous {output_path}')

with open ("crf_model_bio_predictions.txt", "w") as file:
    for i in range(len(y_pred)):
        for j in range(len(y_test[i])):
            file.write(f"Token: {X_test[i][j]} | Predicted label: {y_pred[i][j]}")

print("Prédictions enregistrées dans crf_model_bio_predictions.txt")
