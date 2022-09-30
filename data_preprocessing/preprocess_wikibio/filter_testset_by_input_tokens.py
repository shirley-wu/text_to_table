import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('oup', nargs="+")
    parser.add_argument('--filter-by-n-token', default=1024, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.inp) as f:
        to_keep = [len(x.strip().split(" ")) <= args.filter_by_n_token for x in f]

    for fname in fnames:
        with open(fname) as f:
            lines = f.readlines()

        os.rename(fname, fname + '.backup-before-filtering')

        with open(fname, 'w') as f:
            for keep, line in zip(to_keep, lines):
                if keep:
                    f.write(line.strip() + '\n')
