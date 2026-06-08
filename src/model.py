from transformers import AutoModelForTokenClassification


def load_model(
    model_name,
    labels_list,
    id2label,
    label2id
):

    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels_list),
        id2label=id2label,
        label2id=label2id
    )
