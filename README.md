## Introduction

This repository is for our ACL2022 paper: [Text-to-Table: A New Way of Information Extraction](https://arxiv.org/abs/2109.02707).

## Requirements

Training requires `fairseq==v0.10.2`, and evaluation requires `sacrebleu==v2.0.0 bert_score==v0.3.11`

Or you can directly install by `pip install -r requirements.txt`.

Note: to avoid potential incompatibility, your fairseq version should be **exactly v0.10.2**, and your python version should be **<3.9**

## Dataset

You can download the four datasets from [Google Drive](https://drive.google.com/file/d/1zTfDFCl1nf_giX7IniY5WbXi9tAuEHDn/view?usp=sharing). If you are interested in preprocessing original table-to-text datasets into our text-to-table datasets, please check [data_preprocessing](data_preprocessing/).

For preprocessing, we use `fairseq` for BPE and binarization. You need to first download a BART model [here](https://github.com/pytorch/fairseq/tree/main/examples/bart), and then use `scripts/preprocess.sh` to preprocess the data. The script has two arguments: the first is the data path and the second is the bart model path, e.g.,
```bash
bash scripts/preprocess.sh data/rotowire/ bart.base/
```
then you'll have BPE-ed files under `data/rotowire` and binary files under `data/rotowire/bins`.

## Training

For each dataset, use `scripts/dataset-name/train_vanilla.sh` to train a vanilla seq2seq model, and use `scripts/dataset-name/train_vanilla.sh` to train a HAD model. The training scripts have two arguments: the first is the data path (NOTE: it's not the path to the binary files) and the second is the bart model path, e.g.,
```bash
bash scripts/rotowire/train_had.sh data/rotowire/ bart.base/
```

Additionally, for Rotowire and WikiTableText, the datasets are very small, so we run experiments with 5 seeds (1, 10, 20, 30, 40) and report the average numbers. Scripts under `scripts/rotowire` and `scripts/wikitabletext` have the seed as the third argument.

Rotowire and WikiBio experiments are run on 8 GPUs. E2E and WikiTableText experiments are run on 1 GPU.

You'll need GPUs that supports `--fp16` (such as V100). If not, please remove the `--fp16` option in the scripts.

## Inference and Evaluation

For each dataset, use `scripts/dataset-name/test_vanilla.sh` to test with vanilla decoding, and use `scripts/dataset-name/test_constraint.sh` to test with table constraint. The test scripts have two arguments: the first is the data path and the second is the checkpoint path (by default it is where your saved checkpoint goes to), e.g.,
```bash
bash scripts/rotowire/test_constraint.sh data/rotowire/ 
```

Similar to training, you'll need GPUs that supports `--fp16`. If not, please remove `--fp16` in the script.
