import numpy as np
import pickle


def save(path, embs, ids):
  embs_list = list(embs)
  assert len(embs_list) == len(ids)
  tmp = list(zip(ids, embs_list))
  print(f"Saving to {path}")
  with open(path, "wb") as f:
    pickle.dump(tmp, f)


def load(path):
  with open(path, "rb") as f:
    tmp = pickle.load(f)
  _ids = [elem[0] for elem in tmp]
  embs = np.array([elem[1] for elem in tmp])
  return _ids, embs
