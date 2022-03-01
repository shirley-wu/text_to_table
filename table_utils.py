import numpy as np

SEP = "|"


def parse_table_to_text(table, one_line=False, escape_token=None):
    table = [[str(xx) for xx in x] for x in table]
    has_sep_inside = any([SEP in xx for x in table for xx in x])
    if escape_token is None:
        assert not has_sep_inside
    elif has_sep_inside:
        # print("Escape! '{}' -> '{}'. Data: {}".format(SEP, escape_token, table))
        table = [[xx.replace(SEP, escape_token) for xx in x] for x in table]
    text = "\n".join([("{} ".format(SEP) + " {} ".format(SEP).join(x) +
                       " {}".format(SEP)) for x in table])
    if one_line:
        text = text.replace("\n", " <NEWLINE> ")
    return text


def parse_text_to_table(text, strict=False):
    text = text.replace(" <NEWLINE> ", "\n").strip()
    data = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(SEP):
            line = SEP + line
        if not line.endswith(SEP):
            line = line + SEP
        data.append([x.strip() for x in line[1:-1].split(SEP)])
    if not strict and len(data) > 0:
        n_col = len(data[0])
        data = [d[:n_col] for d in data]
        data = [d + ["", ] * (n_col - len(d)) for d in data]
    try:
        data = np.array(data, dtype=np.str)
    except:
        assert strict
        data = None
    return data


def extract_table_by_name(text, name):
    lines = [line.strip() for line in text.replace(" <NEWLINE> ", "\n").strip().splitlines()]
    if name + ":" not in lines:
        return ""
    table = []
    for line in lines[lines.index(name + ":") + 1:]:
        if line.endswith(":"):
            break
        table.append(line.strip())
    return " <NEWLINE> ".join(table)


def is_empty_table(table, row_name: bool, col_name: bool):
    if table is None:
        return True
    if len(table.shape) != 2:
        assert table.size == 0
        return True
    row, col = table.shape
    if row_name and col < 2:
        return True
    if col_name and row < 2:
        return True
    return False
