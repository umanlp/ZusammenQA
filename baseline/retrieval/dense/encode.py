import tqdm
import numpy as np

from itertools import islice
from sentence_transformers import SentenceTransformer

_checkpoints_basedir = ""
models = {
  "distildmbert": _checkpoints_basedir + "public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_distilbert-multilingual-nli-stsb-quora-ranking.zip",
  "distilxlmr": _checkpoints_basedir + "sbert.net_models_xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
  "distiluse": "distiluse-base-multilingual-cased-v2",
  "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
  "mpnet": "paraphrase-multilingual-mpnet-base-v2",
  "labse": "LaBSE"
}

_model_cache = None
def encode_distil_x(lines, model):
  global _model_cache
  _path = models[model]
  if not _model_cache:
    embedder = SentenceTransformer(_path)
    embedder.max_seq_length = 128
    embedder.eval()
    embedder = embedder.cuda()
    _model_cache = embedder
  else:
    embedder = _model_cache
    embedder = embedder.cuda()
  batches = list(chunk(lines, size=100))
  embeddings = []
  for batch in tqdm.tqdm(batches):
    embeddings.extend(
      embedder.encode(
        batch, 
        show_progress_bar=False,
        convert_to_tensor=False,
        convert_to_numpy=True,
        # is_pretokenized=False,
        output_value="sentence_embedding"
      )
    )
  embeddings = np.array(embeddings)
  embedder.cpu()
  return embeddings


def chunk(it, size):
  it = iter(it)
  return iter(lambda: tuple(islice(it, size)), ())
