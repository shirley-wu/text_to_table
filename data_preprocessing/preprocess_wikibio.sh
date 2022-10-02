# Download data: please do this manually.
# We use the preprocessed version released by "Enhancing Neural Data-To-Text Generation Models with External Background Knowledge (Chen et al., EMNLP, 2019)"
# Please manually download WikiInfo2Text.zip from https://github.com/hitercs/WikiInfo2Text and unzip it under raw/
# We'll use raw/WikiInfo2Text/WikiBio/

# Preprocess text
mkdir -p preprocessed/wikibio
python -c "import nltk; nltk.download('stopwords')"
PYTHONPATH=.:../ python ./preprocess_wikibio/preprocess_wikibio.py raw/WikiInfo2Text/WikiBio/ preprocessed/wikibio/

# Note: after preprocessing, we also filter the test set to remove too long input documents.
# To do so, first conduct BPE (L1-L24 in scripts/preprocess.sh), and then do filtering via
# python ./preprocess_wikibio/filter_testset_by_input_tokens.py --inp preprocessed/wikibio/test.bpe.text --oup preprocessed/wikibio/test.bpe.text preprocessed/wikibio/test.bpe.data preprocessed/wikibio/test.text.detok preprocessed/wikibio/test.data
