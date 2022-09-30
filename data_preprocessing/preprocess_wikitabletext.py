import os
import string
import sys

from nltk.corpus import stopwords

from detok_utils import detokenize
from table_utils import parse_table_to_text


def convert_line_to_data_and_text(line):
    _, k, v, sent = line.strip().split("\t")
    # data
    k = k.replace("_$$_", " ").split("_||_")
    assert k[0] == "subj_title"
    k[0] = "title"
    assert k[1] == "subj_subtitle"
    k[1] = "subtitle"
    v = v.replace("_$$_", " ").split("_||_")
    data = [[kk, vv] for kk, vv in zip(k, v)]
    # text
    sent = sent.replace("_$$_", " ")
    return data, sent


def is_empty(x):
    if x.strip().replace(" ", "") == "n/a":
        return True
    if not any(t in string.ascii_letters + string.digits for t in x):
        return True
    return False


def is_counted_as_token(token):
    if not all(t in string.ascii_letters + string.digits + '.' for t in token):
        return False
    if all(t in string.ascii_letters for t in token) and len(token) == 1:
        return False
    if token in stopwords.words('english'):
        return False
    return True


def is_value_hallucination(value, text):
    vtokens = [vv for vv in value.split() if is_counted_as_token(vv)]
    if len(vtokens) == 0:
        vtokens = value.split()
    return any(vv not in text for vv in vtokens)


def filter_data(data, text):
    assert data[0][0] == "title"
    title = data[0][1]
    assert data[1][0] == "subtitle"
    subtitle = data[1][1]

    data = {k: v for k, v in data[2:]}

    # remove some keys and attributes
    data = {k: v for k, v in data.items() if not is_empty(v)}

    # check tokens in text
    for k in list(data.keys()):
        if is_value_hallucination(data[k], text):
            data.pop(k)

    # add title & subtitle back to the table
    data = [("subtitle", subtitle), ] + [(k, v) for k, v in data.items()]
    if not is_value_hallucination(title, text):
        data = [("title", title), ] + data
    return data


if __name__ == "__main__":
    _, inp_suffix, oup_dir = sys.argv

    for inp_split, oup_split in [('train', 'train'), ('dev', 'valid'), ('test', 'test'), ]:
        with open(inp_suffix + '.' + inp_split) as f:
            lines = f.readlines()

        with open(os.path.join(oup_dir, f'{oup_split}.text'), 'w') as ftext, \
                open(os.path.join(oup_dir, f'{oup_split}.data'), 'w') as fdata:
            for line in lines:
                data, text = convert_line_to_data_and_text(line)

                ftext.write(detokenize(text.split(" ")).strip() + '\n')

                data = filter_data(data, text)
                data = [[detokenize(kk.split(" ")), detokenize(vv.split(" "))] for kk, vv in data]
                fdata.write(parse_table_to_text(data, one_line=True) + '\n')
