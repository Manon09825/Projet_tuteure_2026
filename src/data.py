from datasets import Dataset
import random

def load_data(filepath):

    sentences = []
    labels = []

    tokens = []
    tags = []

    end_punctuation = ['.', '!', '?']

    with open(filepath, 'r', encoding='utf-8') as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            token = parts[0]
            label = parts[-1]
            if label == '_':
                label = "O"

            tokens.append(token)
            tags.append(label)

            if token[-1] in end_punctuation:
                sentences.append(tokens)
                labels.append(tags)

                tokens = []
                tags = []

    return sentences, labels


def augment_alt(sentences, tags):

    augmented_sentences = []
    augmented_tags = []

    for sentence, tag in zip(sentences, tags):
        augmented_sentences.append(sentence)
        augmented_tags.append(tag)

        if any("ALT" in label for label in tag):
            for i in range(5):
                augmented_sentences.append(sentence.copy())
                augmented_tags.append(tag.copy())

        else:
            if random.random() < 0.3:
                augmented_sentences.append(sentence.copy())
                augmented_tags.append(tag.copy())

    return augmented_sentences, augmented_tags


def augment_spans(sentences, tags):

    augmented_sentences = []
    augmented_tags = []

    for sentence, tag in zip(sentences, tags):
        augmented_sentences.append(sentence)
        augmented_tags.append(tag)

        if ("SPAN" in label for label in tag):
            for i in range(5):
                augmented_sentences.append(sentence.copy())
                augmented_tags.append(tag.copy())

        else:
            if random.random() < 0.3:
                augmented_sentences.append(sentence.copy())
                augmented_tags.append(tag.copy())

    return augmented_sentences, augmented_tags


def map_labels():
    labels = ['O',
        'B-ALTC', 'I-ALTC',
        'B-ALTE','I-ALTE',
        'B-NONALT', 'I-NONALT',
        'B-ALTS', 'I-ALTS']

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for i, label in enumerate(labels)
    }

    return label2id, id2label


def map_span_labels():
    labels = ['O', 'B-SPAN', 'I-SPAN']

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for i, label in enumerate(labels)
    }

    return label2id, id2label



def convert_labels_to_ids(labels, label2id):

    return [
        [label2id[l] for l in sent]
        for sent in labels
    ]


def make_dataset(sentences, label_ids):

    return Dataset.from_dict({
        "tokens": sentences,
        "tags": label_ids
    })
