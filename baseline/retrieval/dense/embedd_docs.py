import csv
import os 
import tqdm
import argparse
from encode import encode_distil_x, models
from load_safe import save


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=list(models.keys()))
parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

path_corpus = "data/corpora/pkl/"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def load_corpus(limit=-1):
    dids, docs = [], []
    empty_lines = 0
    with open("mDPR/mia2022_shared_task_all_langs_w100.tsv") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: doc_id, doc_text, title
        for i, row in tqdm.tqdm(enumerate(reader), total=46540290):
            if row[0] != 'id':
                int(row[0])
                d_id = row[0]
                title = row[2]
                text = row[1]
                dids.append(d_id)
                docs.append(" ".join([title, text]))
                if i == limit:
                    break
    print(f"{empty_lines} empty lines")
    return dids, docs


def embedd_corpus(model):
    tgt_dir = f"retrieval/model={model}/"
    print(f"target directory: {tgt_dir}")
    os.makedirs(tgt_dir, exist_ok=True)
    tgt_file = tgt_dir + "corpus.pkl"
    dids, docs = load_corpus()
    print("Corpus loaded")
    d_embs = encode_distil_x(docs, model)
    save(path=tgt_file, ids=dids, embs=d_embs)


if __name__ == '__main__':
    print(f"using gpu {args.gpu}")
    embedd_corpus(args.model)
