import csv
import os
import sys

from table_utils import parse_table_to_text


def fix_key(x):
    x_ = ""
    for xx in x:
        if xx.isupper():
            x_ += " " + xx.lower()
        else:
            x_ += xx
    return x_.strip()


def convert_mr_to_table(mr):
    mr = mr.split(",")
    table = []
    for x in mr:
        k = fix_key(x.split("[")[0].strip()).capitalize()
        v = x.split("[")[1].split("]")[0].strip().capitalize()
        table.append([k, v])
    return table


def read_dataset(fname):
    with open(fname) as f:
        lines = list(csv.reader(f))
    assert lines[0] == ['mr', 'ref', ]
    return [(convert_mr_to_table(mr), text) for mr, text in lines[1:]]


def filter_data(text, data):
    text_ = text.lower()
    filtered_data = []
    for k, v in data:
        if k == "Name":  # ** almost ** all names are correct. No need to filter anything
            filtered_data.append((k, v))
        elif k == "Eat type":  # value choices: {'Coffee shop', 'Restaurant', 'Pub'}
            pass
        elif k == "Price range":  # {'High', 'More than £30', 'Moderate', '£20-25', 'Less than £20', 'Cheap'}
            if "price" in text_ or "pricing" in text_ or "cost" in text_ or "afford" in text_ or \
                    "£" in text or "pound" in text_ or "cheap" in text_ or "expensive" in text_:
                filtered_data.append((k, v))
        elif k == "Customer rating":  # {'3 out of 5', 'High', 'Low', 'Average', '5 out of 5', '1 out of 5'}
            if "rate" in text_ or "rating" in text_ or "review" in text_:
                filtered_data.append((k, v))
            elif "out of" in v and v.split("out of")[0].strip() in text_:
                filtered_data.append((k, v))
            elif v.lower() in text_:  # allowing directly using 'high' 'low' 'average' to describe it
                filtered_data.append((k, v))
        elif k == "Near":
            if v.lower() in text_ or "near" in text_ or "close" in text_:
                filtered_data.append((k, v))
        elif k == "Food":  # {'French', 'Chinese', 'Fast food', 'Japanese', 'Indian', 'Italian', 'English'}
            if v.lower() in text_ or (v == "Fast food" and "fast" in text_):
                filtered_data.append((k, v))
        elif k == "Area":  # {'City centre', 'Riverside'}
            if "area" in text_ or v.lower() in text_:
                filtered_data.append((k, v))
            elif v == "City centre" and ("city" in text_ or "center" in text_ or "centre" in text_):
                filtered_data.append((k, v))
            elif v == "Riverside" and "river" in text_:
                filtered_data.append((k, v))
        elif k == "Family friendly":  # {'No', 'Yes'}
            if "family" in text_ or "families" in text_ or "friendly" in text_ or \
                    "child" in text_ or "kid" in text_ or "adult" in text_:
                filtered_data.append((k, v))
        else:
            raise KeyError(k)
    return filtered_data


if __name__ == "__main__":
    _, inp_dir, oup_dir = sys.argv

    for inp_split, oup_split in [('trainset', 'train'), ('devset', 'valid'), ('testset_w_refs', 'test'), ]:
        dataset = read_dataset(os.path.join(inp_dir, inp_split + '.csv'))
        with open(os.path.join(oup_dir, f'{oup_split}.text'), 'w') as ftext, \
                open(os.path.join(oup_dir, f'{oup_split}.data'), 'w') as fdata:
            for table, text in dataset:
                filtered_table = filter_data(text, table)
                ftext.write(text.replace("\n", " <NEWLINE> ") + '\n')
                fdata.write(parse_table_to_text(filtered_table, one_line=True).strip() + '\n')
