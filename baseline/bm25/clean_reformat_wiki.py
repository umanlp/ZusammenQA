import os
import json
from tqdm import tqdm
from collections import defaultdict

path = "/path/to/mia_2022_train_data.jsonl"
with open(path, "r") as f:
  lines = [l for l in f.readlines()]
  
records = [json.loads(l) for l in lines]
lang2queries = defaultdict(list)
lang2queries_gold = defaultdict(list)

for r in tqdm(enumerate(records)):
  language = r['lang']
  question_ = " ".join(r['question'].replace("\t", " ").split())
  lang2queries[language].append(f"{r['id']}\t{question_}\n")
  answers = " ".join(r['answers'])
  answers = " ".join(answers.split())
  lang2queries_gold[language].append(f"{r['id']}\t{question_} {answers}\n")

for k, v in lang2queries_gold.items():
  print(f"{k}: {len(v)}")
  
os.makedirs("data/queries/", exist_ok=True)
for lang in tqdm(lang2queries.keys()):
  with open(f"data/queries/{lang}.tsv", "w") as f:
    f.writelines(lang2queries[lang])
  with open(f"data/queries/{lang}_gold.tsv", "w") as f:
    f.writelines(lang2queries_gold[lang])
