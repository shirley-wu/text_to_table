import json
import os
import sys
import tqdm

from detok_utils import detokenize

if __name__ == "__main__":
    _, inp_dir, oup_dir = sys.argv

    for split in ['train', 'valid', 'test', ]:
        with open(os.path.join(inp_dir, f'{split}.json'))as f:
            o = json.load(f)
        with open(os.path.join(oup_dir, f'{split}.text'), 'w') as f:
            for oo in tqdm.tqdm(o, desc=split):
                f.write(detokenize(oo['summary']) + '\n')
