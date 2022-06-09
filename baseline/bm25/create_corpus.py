import logging
import os
import json

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

languages = [
  # # Training 
  'en', 'ar', 'bn', 'fi', 'ja', 'ko', 'ru', 'te',
  # Test
  'km', 'ms', 'sv', 'tr', 'zh_cn', 'es'
  # Surprise
  'tl', 'ta'
]

if __name__ == '__main__':
  wikipedia_dir = "data/wikipedia"
  corpus_dir = "data/corpora"
  for language in languages:
    tgt_dir = os.path.join(corpus_dir, language)
    os.makedirs(tgt_dir, exist_ok=True)
  
    tgt_file = os.path.join(tgt_dir, f"{language}.jsonl")
    if not os.path.exists(tgt_file):
      logger.info(f"creating corpus file for {language}")
      records = []
  
      logger.info("Collecting records")
      lang_dir = os.path.join(wikipedia_dir, language)
      files_and_folders = list(os.walk(lang_dir))
      for dirpath, dirnames, filenames in tqdm(files_and_folders):
        for filename in filenames:
          if filename.startswith("wiki_"):
            with open(os.path.join(dirpath, filename)) as f:
              records.extend(f.readlines())
  
      logger.info("Filtering empty records")
      pyserini_records = []
      for r in tqdm(records):
        r = eval(r)
        if r['text']:
          pyserini_records.append(
            json.dumps({'id': r['id'], 'contents': r['text']}) + "\n"
          )
      logger.info(f"Number of non-empty articles: {len(pyserini_records)}")
      logger.info(f"Storing aggregated records to {tgt_file}")
      with open(tgt_file, "w") as f:
        f.writelines(pyserini_records)
    else:
      logger.info(f"Skipping {language}, file exists ({tgt_file})")
  logger.info("Done")
