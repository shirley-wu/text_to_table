import argparse
import json
import multiprocessing
import os
import string

from nltk.corpus import stopwords

from detok_utils import detokenize
from table_utils import parse_table_to_text


def is_counted_as_token(token):
    if not all(t in string.ascii_letters + string.digits + '.' for t in token):
        return False
    if all(t in string.ascii_letters for t in token) and len(token) == 1:
        return False
    if token in stopwords.words('english'):
        return False
    return True


def is_blocked_key(key):
    if key in {"updated", "", }:
        return True
    if "image" in key or "caption" in key:
        return True
    return False


def is_empty(x):
    if x.strip().replace(" ", "") == "n/a":
        return True
    if not any(t in string.ascii_letters + string.digits for t in x):
        return True
    return False


def filter_data(data, text):
    text_tokens = set(text.split(" "))

    # remove some keys and attributes
    data = {k: v for k, v in data if not is_blocked_key(k) and not is_empty(v)}

    # name and its alias
    if 'name' not in data:
        assert "article_title" in data
        data['name'] = data['article_title']
    if "article_title" in data:
        data.pop("article_title")
    for k in list(data.keys()):
        if 'name' in k and k != 'name':
            if data[k] == data['name']:
                data.pop(k)
            else:
                vtokens = data[k].split()
                if any(vv not in text for vv in vtokens):
                    data.pop(k)

    # check tokens in text
    for k in list(data.keys()):
        if k == "name":  # please don't filter out name
            continue
        v = data[k]
        vtokens = [vv for vv in v.split() if is_counted_as_token(vv)]
        if len(vtokens) == 0:
            vtokens = v.split()
        if any(not (vv in text if len(vv) >= 5 else vv in text_tokens) for vv in vtokens):
            # for longer word (length >= 5), allow text matching
            # for short word (lengths < 5), force token-level matching
            data.pop(k)

    return [(k.replace("_", " "), v) for k, v in data.items()]


def func(jline):
    # parse input
    o = json.loads(jline.strip())
    text = ' '.join(o['Sentences'])
    data = [[k, v] for k, v in o.items() if
            k not in {'Sentences', 'KB_id_tuples', 'KB_str_tuples'} and v != "<none>"]

    # filter and process data
    data = filter_data(data, text)
    data = [(detokenize(k.split(" ")), detokenize(v.split(" "))) for k, v in data]
    data = parse_table_to_text(data, escape_token="", one_line=True)

    return data, detokenize(text.split(" "))


def main(inp_dir, oup_dir, map_fn):
    for inp_split, oup_split in [('train', 'train'), ('dev', 'valid'), ('test', 'test'), ]:
        n = 0
        with open(os.path.join(inp_dir, f'{inp_split}.json')) as fj:
            with open(os.path.join(oup_dir, f'{oup_split}.text'), 'w') as ftext, \
                    open(os.path.join(oup_dir, f'{oup_split}.data'), 'w') as fdata:
                for data, text in map_fn(func, fj):
                    fdata.write(data.strip() + '\n')
                    ftext.write(text.strip() + '\n')
                    n += 1
                    if n % 10000 == 0:
                        print("Processed %d lines" % n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_dir')
    parser.add_argument('oup_dir')
    parser.add_argument('--n-proc', default=16, type=int)
    args = parser.parse_args()

    if args.n_proc == 0:
        main(args.inp_dir, args.oup_dir, map)
    else:
        with multiprocessing.Pool(args.n_proc) as p:
            main(args.inp_dir, args.oup_dir, p.imap)
