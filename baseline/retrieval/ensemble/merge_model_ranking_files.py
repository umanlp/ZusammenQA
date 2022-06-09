# This script reads for each model the ranking files, the folder structure is expected as follows:
# /retrieval_results/{distildmbert,distilxlmr,labse,distilmuse,mpnet,minilm}
# ├── mkqa_dev
# │       ├── mkqa-ar.jsonl
# │       ├── mkqa-ar_xor_retrieve_results.json
# │       ├── mkqa-en.jsonl
# │       ├── mkqa-en_xor_retrieve_results.json
# │       ├── mkqa-es.jsonl
# │       ├── mkqa-es_xor_retrieve_results.json
# │       ├── mkqa-fi.jsonl
# │       ├── mkqa-fi_xor_retrieve_results.json
# │       ├── mkqa-ja.jsonl
# │       ├── mkqa-ja_xor_retrieve_results.json
# │       ├── mkqa-km.jsonl
# │       ├── mkqa-km_xor_retrieve_results.json
# │       ├── mkqa-ko.jsonl
# │       ├── mkqa-ko_xor_retrieve_results.json
# │       ├── mkqa-ms.jsonl
# │       ├── mkqa-ms_xor_retrieve_results.json
# │       ├── mkqa-ru.jsonl
# │       ├── mkqa-ru_xor_retrieve_results.json
# │       ├── mkqa-sv.jsonl
# │       └── mkqa-sv_xor_retrieve_results.json
# ├── retrieve.sh
# └── xorqa
#     ├── mia_2022_dev_xorqa.jsonl
#     ├── mia_2022_dev_xorqa_xor_retrieve_results.json
#     ├── mia_2022_test_xorqa_without_answers.jsonl
#     └── mia_2022_test_xorqa_without_answers_xor_retrieve_results.json
# 
# and writes results (with the same folder structure) to:
# retrieval_results/{ensemble_rank,ensemble_score}

import os
import json
import tqdm
import pickle
import multiprocessing
from copy import copy

from baseline.retrieval.ensemble.info import datasets, models
from baseline.mDPR.dpr.data.qa_validation import has_answer
from baseline.mDPR.dpr.utils.tokenizers import SimpleTokenizer
from baseline.retrieval.ensemble.utils import load_model_rankings
from collections import defaultdict

ensemble_rank_based = True
dry_run = False

if "bm25" in models:
  # no score-based ensembling if bm25 is used, because of scale mismatch between cosine-score and bm25-score 
  assert ensemble_rank_based

ensemble_model_str = "rank-based" if ensemble_rank_based else "score-based"

suffix = "_+mdpr" #"_tmp"
rank_str = "ensemble_rank%s" % suffix
score_str = "ensemble_score%s" % suffix
print(f"Running ensemble ({ensemble_model_str})")

weights_file = "retrieval_results/weights%s.pkl" % suffix
do_weights = os.path.exists(weights_file)


def process_file(file):
  try:
    print(f"Start processing {file}")
    lang2model2avg_score = defaultdict(lambda: defaultdict(list))
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)
    
    model2contexts = load_model_rankings(file)
    ref = model2contexts[list(model2contexts.keys())[0]]
    n_queries = len(ref)
    
    lang2model2weights = None
    # if os.path.exists(weights_file) and do_weights:
    #   with open(weights_file, "rb") as f:
    #     lang2model2weights = pickle.load(f)
    
    query_results = []
    print(f"Process {n_queries} queries")
    for i in tqdm.tqdm(list(range(n_queries))):
      qid_field = 'q_id' if 'q_id' in ref[i] else 'id'
      qid = ref[i][qid_field]
      
      question = ref[i]['question']
      answers = ref[i]['answers'] if 'answers' in ref[i] else ""
      lang = ref[i]['lang']
      
      # collect rankings and scores from each model
      weights_mapping = lang2model2weights[lang] if lang2model2weights else defaultdict(lambda: 1)
      all_doc_ids, docid2doc, to_be_merged, missing = collect_doc_scores(i, model2contexts, weights_mapping)
      
      assert len(models) == len(to_be_merged)
      for model, ranking in zip(models, to_be_merged):
        docs_ranking = [docid2doc[doc_id] for i, doc_id in enumerate(ranking.keys()) if doc_id in docid2doc and i <15]
        docs_ranking = [elem["title"] + " " + elem["text"] for elem in docs_ranking]
        lang2model2avg_score[lang][model].append(
          1 if any(
            has_answer(answers, doc, tokenizer, match_type="string") for doc in docs_ranking
          ) else 0
        )
      
      docid_score = merge_rankings(all_doc_ids, to_be_merged)
      
      # rerank newly scored document ids, if ensemble rank: sort scores ascending, if ensemble score: sort descending
      multiplier = 1 if ensemble_model_str == "rank-based" else -1
      new_ranking = sorted(docid_score, key=lambda elem: elem[1]*multiplier)
      
      
      # ensemble_docid2score = {d_id: score for d_id, score in new_ranking}
      # ensemble_docs_ranking = [ensemble_docid2score[doc_id] for i, doc_id in enumerate(ensemble_docid2score.keys()) if doc_id in docid2doc and i < 15]
      # ensemble_docs_ranking = [elem["title"] + " " + elem["text"] for elem in ensemble_docs_ranking]
      # lang2model2avg_score[lang][ensemble_model_str].append(
      #   1 if any(
      #     has_answer(answers, doc, tokenizer, match_type="string") for doc in ensemble_docs_ranking
      #   ) else 0
      # )
      
      
      # Select top-15 documents
      ctxs = []
      for doc_id, score in new_ranking:
        doc = copy(docid2doc[doc_id])
        doc['score'] = score
        ctxs.append(doc)
        if len(ctxs) == 15:
          break
      
      # Assemble new record
      query_result = {
        qid_field: qid,
        'question': question,
        'lang': lang
      }
      if answers: query_result['answers'] = answers
      query_result['ctxs'] = ctxs
      query_results.append(query_result)
    
    # convert to dict for pickle/multiprocessing
    lang2model2avg_score = {
      lang: {
        # model: sum(scores)/len(scores) if len(scores) > 0 else 0 for model, scores in model2scores.items()
        model: scores for model, scores in model2scores.items()
      } for lang, model2scores in lang2model2avg_score.items()
    }
    return file, query_results, lang2model2avg_score
  except BaseException as e:
    print(e)
    return -1, -1, -1


def merge_rankings(all_doc_ids, to_be_merged):
  # default score is used when one document is present in one results list but not in the other
  # rank-based ensembling: default rank = max + 1
  # score-based ensembling: default score = 0 (equals to worst score in the context of cosine and dot-product)
  default_score = len(to_be_merged[0]) + 1 if ensemble_rank_based else 0
  # re-score each document id
  docid_score = []
  for docid in all_doc_ids:
    scores = []
    for ranking in to_be_merged:
      if docid in ranking:
        scores.append(ranking[docid])
      else:
        scores.append(default_score)
    ensemble_score = sum(scores) / len(scores)
    docid_score.append((docid, ensemble_score))
  return docid_score


def collect_doc_scores(query_idx, model2contexts, model2weights):
  all_doc_ids = set()
  to_be_merged = []
  doci2doc = {}
  
  empty = set()
  for model in models:
    ranking = model2contexts[model][query_idx]['ctxs']
    
    # collect (a) rank-based scores, (b) actual model scores
    if ensemble_rank_based:
      did2score = {elem['id']: float(i) * model2weights[model] for i, elem in enumerate(ranking, 1)}
    else:
      did2score = {elem['id']: float(elem['score']) * model2weights[model] * model2weights[model] for elem in ranking}
    
    for elem in ranking:
      doci2doc[elem['id']] = elem
    
    to_be_merged.append(did2score)
    for did in to_be_merged[-1].keys():
      all_doc_ids.add(did)
  return all_doc_ids, doci2doc, to_be_merged, empty


def main():
  files = []
  for name, dataset in datasets.items():
    for split, ds_files in dataset.items():
      files.extend(ds_files)
  files = [f for f in files if not os.path.exists(
    f % (rank_str if ensemble_rank_based else score_str)
  )]
  
  print(f"Processing the following files: {files}")
  lang2model2scores = defaultdict(lambda: defaultdict(list))
  
  n_cpu = os.cpu_count()
  with multiprocessing.Pool(processes=n_cpu // 2) as pool:
    for file, query_results, single_lang2model2scores in pool.imap_unordered(process_file, files):
      
      if file == query_results == single_lang2model2scores == 1:
        print("\n\nLook into file: %s!!!!!\n\n" % file)
        continue
      
      print("Done processing %s" % file)
      for lang, model2scores in single_lang2model2scores.items():
        for model, scores in model2scores.items():
          lang2model2scores[lang][model].extend(scores)
      
      # TODO: implement solution for Tamil and Tagalog, 
      # TODO: implement: all models equal vote, best global model, model that includes ta/tl, lang2vec
      
      save_path = file % (rank_str if ensemble_rank_based else score_str)
      save_dir = os.path.dirname(save_path)
      if not os.path.exists(save_path):
        os.makedirs(save_dir, exist_ok=True)
        from pathlib import Path
        print(f"creating directory: {save_dir}")
        with open(os.path.join(Path(save_dir).parent, "README.md"), "w") as f:
          f.write(f"Participating ensemble models: {models}")
        print(f"Writing results to: {save_path}", end=" ")
        with open(save_path, "w") as f:
          json.dump(query_results, f)
      
      print(f"Done with {file}")
  
  
  languages = list(lang2model2scores.keys())
  used_models = list(lang2model2scores[languages[0]].keys())
  lang2model2avg_score = defaultdict(dict)
  header = "\t".join(languages)
  lines = [header]
  for model in used_models:
    lang2avg_score =  {
      lang: sum(lang2model2scores[lang][model])/len(lang2model2scores[lang][model]) if len(scores) > 0 else 0 for lang in languages
    }
    for l, avg_score in lang2avg_score.items():
      lang2model2avg_score[l][model] = avg_score
    
    line = "\t".join([model] + [str(round(lang2avg_score[lang], 4)) for lang in languages])
    lines.append(line)
  for line in lines:
    print(line)
  
  with open(weights_file, "wb") as f:
    pickle.dump(lang2model2avg_score, f)


if __name__ == '__main__':
  main()