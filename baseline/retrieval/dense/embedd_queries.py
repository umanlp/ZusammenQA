import argparse
import json
import os
from encode import encode_distil_x
from load_safe import save
from pathlib import Path
from embedd_docs import models


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=list(models.keys()))
parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# process queries
queries_folder = "data/queries/"
os.makedirs(queries_folder, exist_ok=True)
xorqa = {
  "train": [os.path.join(queries_folder, "xorqa", "mia_2022_train_data.jsonl")],
  "dev": [os.path.join(queries_folder, "xorqa", "mia_2022_dev_xorqa.jsonl")],
  "test": [os.path.join(queries_folder, "xorqa", "mia_2022_test_xorqa_without_answers.jsonl"), 
           os.path.join(queries_folder, "xorqa", "mia2022_test_surprise_tamil_without_answers.jsonl"),
           os.path.join(queries_folder, "xorqa", "mia2022_test_surprise_tagalog_without_answers.jsonl"),]
}

fdir = ""
mkqa = {
  "dev": [
    os.path.join(fdir, "mkqa_dev", f"mkqa-{lang}.jsonl")
    for lang in ["fi", "zh_cn", "tr", "ru", "ms", "ko", "sv", "km", "en", "ja", "ar", "es"]
  ],
  "test": [
    os.path.join(fdir, "mkqa_test_without_answers", f"mkqa-{lang}.jsonl")
    for lang in ["pt", "fi", "zh_cn", "tr", "ru", "ms", "ko", "sv", "km", "en", "ja", "ar", "es"]
  ]
}

datasets = {
  # "xorqa": xorqa,
  "mkqa": mkqa,
}

def embedd_queries(model):
  tgt_dir = f"retrieval/model={model}/"
  os.makedirs(tgt_dir, exist_ok=True)
  
  for name, dataset in datasets.items():
    print(name)
    for split in ["train", "dev", "test", "test-surprise"]:
      if split not in dataset:
        continue
        
      for file in dataset[split]:
        print(file)
        fName = Path(file).name.split(".")[0] # remove file extension
        fDir = Path(file).parent.name
        
        tgt_file = os.path.join(tgt_dir, fDir, fName + ".pkl")
        if os.path.exists(tgt_file):
          print(f"skipping {tgt_file}")
          continue
          
        os.makedirs(Path(tgt_file).parent, exist_ok=True)
        qids, queries = [], []
        with open(file, "r") as f:
          for line in f:
            record = json.loads(line)
            qids.append(record["id"])
            queries.append(record["query"] if "query" in record else record["question"])
        q_embs =encode_distil_x(queries, model)
        
        save(path=tgt_file, ids=qids, embs=q_embs)

  print(f"Saved embeddings to: {tgt_dir}")


if __name__ == '__main__':
    print("embedd queries")
    embedd_queries(args.model)
