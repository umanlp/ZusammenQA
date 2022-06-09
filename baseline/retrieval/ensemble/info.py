import os

tldir = "/path/to/retrieval_results_new/%s"

xorqa = {
  
  # (7) ar, bn, fi, ja, ko, ru, te 
  "dev": [os.path.join(tldir, "xorqa", "mia_2022_dev_xorqa.jsonl")],
  
  # (7) ar, bn, fi, ja, ko, ru, te 
  "test": [os.path.join(tldir, "xorqa", "mia_2022_test_xorqa_without_answers.jsonl")],

  # (2) ta, tl
  "test-surprise": [
    os.path.join(tldir, "xorqa", "mia2022_test_surprise_tagalog_without_answers.jsonl"),
    os.path.join(tldir, "xorqa", "mia2022_test_surprise_tamil_without_answers.jsonl"),
  ],

  # (8) ar, bn, en, fi, ja, ko, ru, te
  "train": [os.path.join(tldir, "xorqa", "mia_2022_train_data.jsonl")]
}

mkqa = {
  "dev": [
    os.path.join(tldir, "mkqa_dev", f"mkqa-{lang}.jsonl")
    for lang in ["ar", "en", "es", "fi", "ja", "km", "ko", "ms", "ru", "sv", "tr", "zh_cn"]
    # data/queries/mkqa/mkqa_dev
  ],
  "test": [
    os.path.join(tldir, "mkqa_test_without_answers", f"mkqa-{lang}.jsonl")
    for lang in ["ar", "en", "es", "fi", "ja", "km", "ko", "ms", "ru", "sv", "tr", "zh_cn"]
    # data/queries/mkqa/mkqa_test
  ],
}

datasets = {
  "xorqa": xorqa,
  "mkqa": mkqa,
}

models = [
  # "bm25-oracle",
  "mdpr_baseline_adv_aug_unseen_sur_1e-5_e4_s8269",
  "mdpr_baseline_wiki_en",
  "mdpr_adv_aug_unseen_sur"
  "bm25",
  "distildmbert",
  "distilxlmr",
  "distiluse",
  "minilm",
  "labse",
  "mpnet",
]

