# Download data
mkdir raw
git clone https://github.com/tuetschek/e2e-dataset raw/e2e

# Preprocess
mkdir -p preprocessed/e2e
PYTHONPATH=../ python ./preprocess_e2e.py raw/e2e/ preprocessed/e2e/