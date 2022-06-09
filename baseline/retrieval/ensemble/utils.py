import json

import os

from baseline.retrieval.ensemble.info import models


def load_model_rankings(file):
  model2contexts = {}
  for model in models:
    results_path = file % model
    results_path = results_path.replace(".jsonl","__monolingual_bm25_results.jsonl") if "bm25" in results_path else results_path
    if not os.path.exists(results_path):
      results_path = results_path.replace(".jsonl", "_results.json")
    assert os.path.exists(results_path), f"File not found: {results_path}"
    if True: #not dry_run:
      print(f"Loading rankings for {model}: {results_path}")
      # model2path[model] = results_path
      with open(results_path, "r") as f:
        model2contexts[model] = json.load(f)
  return model2contexts
