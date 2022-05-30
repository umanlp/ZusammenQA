import argparse
import json
import os
import random

import jsonlines
from tqdm import tqdm


def read_json(path):
    data = []
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def read_jsonlines(path):
    data = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data


def main(args):
    assert os.path.exists(args.input_dir)
    files = sorted(
        [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    )

    if args.output_file:
        if not os.path.dirname(args.output_file):
            os.makedirs(os.path.dirname(args.output_file))

    if args.passages_path:
        assert os.path.exists(args.passages_path)
        passages = read_json(args.passages_path)
        # passages = list(passages.values())
        print(f"Loaded {len(passages)} total passages.")

    train_pairs = []

    for pair_file in files:
        print("Processing file:", pair_file)
        assert os.path.exists(pair_file)
        data = read_jsonlines(pair_file)

        lang = os.path.basename(pair_file).split(".")[0].split("_")[-1]

        if args.random_n:
            rand_idxs = [random.randint(0, len(data) - 1) for i in range(args.random_n)]
            data = [data[i] for i in rand_idxs]

        for i, d in enumerate(tqdm(data)):
            q = d["question"]
            a = d["answer"]
            # t = d["type"]
            q_id = d["q_id"]
            passage = d["passage"]

            negative_ctx_ids = d["negative_ctx_ids"]

            negative_ctxs = []
            for nid in negative_ctx_ids:
                if nid in passages:
                    negative_ctxs.append(passages[nid])

            train_pair = {
                "lang": lang,
                "question": q,
                "answers": [a],
                "positive_ctxs": [passage],
                "hard_negative_ctxs": [],
                "negative_ctxs": negative_ctxs,
                "q_id": q_id,
            }
            train_pairs.append(train_pair)

        print(f"Generated {len(train_pairs)} training pairs.")
        if args.shuffle:
            random.shuffle(train_pairs)

        if args.output_file:
            with jsonlines.open(args.output_file, "a") as writer:
                writer.write_all(train_pairs)
            print(f"Saved output file to: {args.output_file}")

            train_pairs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", help="Input directory containing QA pair files."
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="./train_data.jsonl",
        type=str,
        help="Output file.",
    )
    parser.add_argument("-p", "--passages-path", type=str, help="Passages path.")
    parser.add_argument(
        "-nn", "--n-negatives", default=5, type=int, help="Number of negative contexts."
    )
    parser.add_argument(
        "-rn", "--random-n", type=int, help="Random number of subsamples."
    )
    parser.add_argument(
        "-ss", "--shuffle", help="Shuffle output file.", action="store_true"
    )
    args = parser.parse_args()

    main(args)
