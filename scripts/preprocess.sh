# Script adopted from https://github.com/pytorch/fairseq/blob/main/examples/bart/README.summarization.md
# Run this script under root directory
# You should download and unzip BART files in advance.
# See https://github.com/pytorch/fairseq/tree/main/examples/bart for BART download links
DATA_DIR=$1
BART_DIR=$2

P=`pwd`/scripts/multiprocessing_bpe_encoder.py
BART_DIR=$( realpath $BART_DIR )
cd ${DATA_DIR}

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
for SPLIT in train valid test; do
  for LANG in text data; do
    python $P \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$SPLIT.$LANG" \
      --outputs "$SPLIT.bpe.$LANG" \
      --workers 60 \
      --keep-empty;
  done
done

fairseq-preprocess --source-lang text --target-lang data \
  --trainpref "train.bpe" --validpref "valid.bpe" --testpref "test.bpe" \
  --destdir "bins/" --workers 60 \
  --srcdict ${BART_DIR}/dict.txt --tgtdict ${BART_DIR}/dict.txt
