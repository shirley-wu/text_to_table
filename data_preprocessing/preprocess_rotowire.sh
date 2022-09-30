# Download data
mkdir raw
git clone https://github.com/harvardnlp/boxscore-data raw/rotowire && cd raw/rotowire && tar -xf rotowire.tar.bz2 && cd ../../

mkdir -p preprocessed/rotowire

# Preprocess text
PYTHONPATH=. python ./preprocess_rotowire/preprocess_text.py raw/rotowire/rotowire preprocessed/rotowire/

# Preprocess data
python -c "import nltk; nltk.download('punkt')"
python ./rotowire/filter_data.py raw/rotowire/rotowire/  # filter data that doesn't appear in text
PYTHONPATH=../ python ./preprocess_rotowire/preprocess_data.py raw/rotowire/rotowire/ preprocessed/rotowire/