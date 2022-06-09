import pathlib
import random
import json
import os.path
from collections import defaultdict

import tqdm
from pyserini.search.lucene import LuceneSearcher
from create_corpus import languages

query_dir = ""
index_dir = ""

do_oracle = False
print(f"do oracle: {do_oracle}")
results_dir = "bm25" + ("_oracle" if do_oracle else "")
batch_size = 10

xorqa_dev = query_dir + "xorqa/mia_2022_dev_xorqa.jsonl"
xorqa_train = query_dir + "xorqa/mia_2022_train_data.jsonl"
xorqa_test = query_dir + "xorqa/mia_2022_test_xorqa_without_answers.jsonl"
xorqa_test_surprise_ta = query_dir + "xorqa/mia2022_test_surprise_tamil_without_answers.jsonl" 
xorqa_test_surprise_tl = query_dir + "xorqa/mia2022_test_surprise_tagalog_without_answers.jsonl"

mkqa_dev = [query_dir + f"mkqa/mkqa_dev/mkqa-{lang}.jsonl" for lang in ["ar", "en", "es", "fi", "ja", "km", "ko", "ms", "ru", "sv", "tr", "zh_cn"]]
mkqa_test = [query_dir + f"mkqa/mkqa_test_without_answers/mkqa-{lang}.jsonl" for lang in ["ar", "en", "es", "fi", "ja", "km", "ko", "ms", "ru", "sv", "tr", "zh_cn"]]

topk = 15
query_files = [
  # xorqa_train, 
  xorqa_test_surprise_ta,
  xorqa_test_surprise_tl,
  xorqa_dev,
  xorqa_test
]
query_files.extend(mkqa_dev)
query_files.extend(mkqa_test)


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")

for query_file in query_files:
  records = []

  filename = query_file.split("/")[-1]
  filedir = os.path.dirname(query_file)
  tgt_dir = os.path.join(
    results_dir, 
    pathlib.Path(filedir).parent.name,
    pathlib.Path(filedir).name
  )
  
  # tgt_dir = filedir.replace("queries", "retrieval_results")
  tgtfile = os.path.join(tgt_dir, filename.replace(".json", "__monolingual_bm25_results.json"))
  if not os.path.exists(tgtfile):
    lang2queries = defaultdict(list)
    with open(query_file, "r") as f:
      for l in f.readlines():
        q = json.loads(l)
        lang2queries[q['lang']].append(q)

    for lang in languages:
      if lang not in lang2queries:
        continue
        
      queries = lang2queries[lang]
      searcher = LuceneSearcher(os.path.join(index_dir, lang))

      offsets = list(range(0, len(queries), batch_size))
      for i in tqdm.tqdm(offsets):
        batch_queries = queries[i:i+batch_size]
        hits = searcher.batch_search(
          queries=[
            q['question'] + " " + " ".join(q['answers']) if do_oracle and bool(q['answers']) else q['question']
            for q in batch_queries
          ],
          qids =[q['id'] for q in batch_queries],
          k=topk,
          threads=20
        )
        for query in batch_queries:
          record = {
            "q_id": query["id"],
            "question": query["question"],
            "answers": query["answers"],
            "lang": query["lang"],
          }
          ctxs = []
          for doc in hits[query['id']]:
            raw_doc = eval(doc.raw)['contents']
            text = tokenizer.decode(tokenizer.encode(raw_doc, truncation=True, max_length=128, padding=False),
                             skip_special_tokens=True)
            ctx = {
              "id": doc.docid,
              "title": "",
              "text": text,
              "score": doc.score,
              "has_answer": str(any(answer in raw_doc for answer in query["answers"])).lower()
            }
            ctxs.append(ctx)
          record["ctxs"] = ctxs
          records.append(record)

    if records:
      os.makedirs(tgt_dir, exist_ok=True)
      print(f"Writing bm25 results to {tgtfile}")
      with open(tgtfile, "w") as f:
        f.write(json.dumps(records))
    else:
      print("No records collected.")
  else:
    print(f"{tgtfile} exists already, skipping.")
    
  