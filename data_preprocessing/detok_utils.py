from sacremoses import MosesDetokenizer


def restore_escape(text):
    text = text.replace('``', '"').replace("''", '"')  # quote
    text = text.replace("-lrb-", "(").replace("-LRB-", "(").replace("-rrb-", ")").replace("-RRB-", ")")  # parentheses
    return text


def detokenize(tokens):
    text = ' '.join(tokens)
    text = restore_escape(text)
    return MosesDetokenizer(lang='en').detokenize(text.split(" "))
