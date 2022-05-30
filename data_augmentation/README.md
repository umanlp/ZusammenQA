# Data Augmentation via Question Generation

This repository contains the code for generating questions from passages. It largely adopts the code by https://github.com/liamdugan/summary-qg.

## Installation

Conda:
```
conda create -n sumqg_env python=3.9.7
conda activate sumqg_env
pip install -r requirements.txt
python -m nltk.downloader punkt
```
venv:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Usage

To generate the data in the same format as reported in the paper, run the following scripts in order:

Generate QA pairs [only EN]:
```
python run_qg_mia_bulk.py -c -i ./data/wikipedia/en -o ./data/wikipedia_aug_new/en
```

Filter generated QA pairs (also maps to negative context ids) [only EN]:
```
python filter_qa_pairs.py -i ./data/wikipedia_aug_v2/en -s -c
```

Translate the filtered QA pairs [all languages]:
```
python translate_pairs.py -i ./filtered_data/qa_pairs.jsonl -o ./filtered_data -ss 100 -s google
```

Convert the QA pairs into the mDPR file format [all languages]:
```
python3 generate_mDPR_data_v2.py -i ./filtered_data/google -o ./filtered_data/train_data.jsonl -p ./filtered_data/trimmed_passages.json -ss
```