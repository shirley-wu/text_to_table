# Download data
mkdir raw
git clone https://github.com/msra-nlc/Table2Text raw/wikitabletext

# Preprocess
mkdir -p preprocessed/wikitabletext
python -c "import nltk; nltk.download('stopwords')"
PYTHONPATH=.:../ python ./preprocess_wikitabletext.py raw/wikitabletext/MSRA_NLC.Table2Text preprocessed/wikitabletext/