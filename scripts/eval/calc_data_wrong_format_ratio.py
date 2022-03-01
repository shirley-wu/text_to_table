import argparse

from table_utils import (
    extract_table_by_name,
    parse_text_to_table,
    is_empty_table,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('tgt')
    parser.add_argument('--row-header', default=False, action="store_true")
    parser.add_argument('--col-header', default=False, action="store_true")
    parser.add_argument('--table-name', default=None)
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


if __name__ == "__main__":
    args = parse_args()

    hyp_data = []
    with open(args.hyp) as f:
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            hyp_data.append(parse_text_to_table(line, strict=True))
    tgt_data = []
    with open(args.tgt) as f:
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            tgt_data.append(parse_text_to_table(line, strict=True))

    empty_tgt = 0
    wrong_format = 0
    for hyp_table, tgt_table in zip(hyp_data, tgt_data):
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            empty_tgt += 1
        elif hyp_table is None:
            wrong_format += 1

    valid_tgt = len(hyp_data) - empty_tgt
    print("Wrong format: %d / %d (%.2f%%)" % (wrong_format, valid_tgt, wrong_format / valid_tgt * 100))
