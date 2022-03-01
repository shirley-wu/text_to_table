import argparse

from fairseq.data import encoders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('oup')
    parser.add_argument('--bpe', default='gpt2')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bpe = encoders.build_bpe(args)
    with open(args.inp) as f:
        lines = f.readlines()
    with open(args.oup, 'w') as f:
        for line in lines:
            f.write(bpe.decode(line.strip()).replace("\n", " <NEWLINE> ") + '\n')
