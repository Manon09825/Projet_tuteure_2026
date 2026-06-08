from transformers import AutoTokenizer

def tokenize(ex, tokenizer):
    tokenized_inputs = tokenizer(
        ex['tokens'],
        truncation = True,
        is_split_into_words = True,
        )

    all_word_ids = []
    labels = []

    for i, label in enumerate(ex['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        all_word_ids.append(word_ids)

        label_id = []

        for word_idx in word_ids:

            if word_idx is None:
                label_id.append(-100)

            else:
                tag = label[word_idx]
                label_id.append(tag)


        labels.append(label_id)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = all_word_ids
    return tokenized_inputs



def tokenize2(ex):
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    tokenized = tokenizer(
        ex["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels=[]

    for i,label in enumerate(ex["tags"]):

        word_ids = tokenized.word_ids(batch_index=i)

        previous=None
        label_ids=[]

        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(-100)

        previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized["labels"]=labels

    return tokenized
