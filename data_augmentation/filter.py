import argparse
import json
import os
import random
import re
import uuid

import jsonlines
import pysbd
import requests
from dateutil.parser import parse
from tqdm import tqdm


def read_jsonlines(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def is_wiki_page(text):
    try:
        text = text.lower().replace(" ", "_")
        result = requests.get(f"https://en.wikipedia.org/wiki/{text}")
        if result.status_code == 200:  # the article exists
            return True
        else:
            return False
    except:
        return False


def is_number(text):
    try:
        float(text)
        return True
    except:
        return False


def is_who_question(text):
    if "who" in text.lower().split(" ")[0]:
        return True
    else:
        return False


def is_how_many_question(text):
    if "how many" in text.lower()[:8]:
        return True
    else:
        return False


def contains_date(text):
    try:
        parse(text, fuzzy=False)
        return True
    except:
        return False


def get_trim(a, text, lang="en", clean=False):
    seg = pysbd.Segmenter(language=lang, clean=clean)
    sents = seg.segment(text)

    for sent in sents:
        if a in sent:
            return sent

    return ""


def trim_text(text, length=100, loc="start"):
    if loc == "end":
        return " ".join(text.split(" ")[-length:])
    elif loc == "mid-start":
        return " ".join(text.split(" ")[length : length * 2])
    elif loc == "mid-end":
        return " ".join(text.split(" ")[-length * 2 : -length])
    else:
        return " ".join(text.split(" ")[:length])


def clean_text(text):
    # not implemented
    return text


def get_trim(sent, passage, length=100):
    sent_tokens = sent.split(" ")
    missing_tokens = length - len(sent_tokens)

    try:
        if missing_tokens > 0:
            hits = re.search(sent, passage)
            start, end = hits.start(), hits.end()

            before_tokens = passage[:start].split(" ")
            after_tokens = passage[end:].split(" ")
            if len(before_tokens) >= missing_tokens:
                new_passage = " ".join(before_tokens[-missing_tokens:] + sent_tokens)
            elif len(after_tokens) >= missing_tokens:
                new_passage = " ".join(sent_tokens + after_tokens[:missing_tokens])
            else:
                extended_tokens = before_tokens + sent_tokens
                missing_tokens = length - len(extended_tokens)
                new_passage_tokens = extended_tokens + after_tokens[:missing_tokens]
                assert len(new_passage_tokens) == length
                new_passage = " ".join(new_passage_tokens)
        else:
            new_passage = sent
    except:
        new_passage = ""

    return new_passage


def aggregate_all(dirs, trim_len):
    passages = {}
    all_pairs = []
    trimmed_passages = {}

    for d in tqdm(dirs):
        files = sorted([os.path.join(d, f) for f in os.listdir(d)])

        for f in tqdm(files):
            data = read_jsonlines(f)

            for d in data:
                _id = d["id"]
                _revid = d["revid"]
                _url = d["url"]
                _title = d["title"]
                _url = d["url"]

                for i, qa in d[
                    "qa_pairs"
                ].items():  # i-th index from the original qa_pairs

                    qa["f"] = f
                    qa["id"] = _id
                    qa["revid"] = _revid

                    if args.clean:
                        qa["passage"] = clean_text(qa["passage"])

                    passage_id = _url + "_" + _revid + "_" + i
                    if passage_id in passages:
                        print("Passage already exists with id:", passage_id)
                        print("Existing passage:", passages[passage_id])
                        print("QA passage:", qa["passage"])
                        assert qa["passage"] == passages[passage_id]
                    else:
                        passages[passage_id] = {"title": _title, "text": qa["passage"]}

                    for qa_pair in qa["qa_pairs"]:
                        a = qa_pair["answer"]
                        qa_pair["answer_sentence"] = get_trim(a, qa["passage"])
                        if qa_pair["answer_sentence"]:
                            passage_text = get_trim(
                                qa_pair["answer_sentence"],
                                qa["passage"],
                                length=trim_len,
                            )
                            qa_pair["passage"] = {"title": _title, "text": passage_text}
                            trimmed_passages[passage_id] = {
                                "title": _title,
                                "text": passage_text,
                            }

                            qa_pair["passage_id"] = passage_id
                            qa_pair["q_id"] = str(uuid.uuid4())
                            all_pairs.append(qa_pair)
    return all_pairs, passages, trimmed_passages


def compute_valid_pairs(all_pairs):
    valid_idxs = []

    for i, pair in enumerate(tqdm(all_pairs)):
        q = pair["question"].lower()
        a = pair["answer"].lower()

        if is_number(a):
            valid_idxs.append((i, "is_num"))
        elif is_who_question(q):
            valid_idxs.append((i, "is_who"))
        elif is_how_many_question(q):
            valid_idxs.append((i, "is_how_many"))
        # elif is_wiki_page(a):
        #     valid_idxs.append((i,"is_wiki"))
        # elif contains_number(a):
        #     valid_idxs.append((i,"has_num"))
        elif contains_date(a):
            valid_idxs.append((i, "has_date"))
        else:
            pass

    return valid_idxs


def get_valid_data(
    valid_idxs,
    all_pairs,
    passages,
    trimmed_passages,
    save_files=True,
    output_dir=None,
    n_negatives=None,
    n_languages=None,
):
    filtered_passages = {}
    filtered_pairs = []

    passage_keys = list(passages.keys())

    for i, t in valid_idxs:
        pair = all_pairs[i]
        pair["type"] = t
        a = all_pairs[i]["answer"]

        passage_id = pair["passage_id"]

        negative_ctx_ids = []
        for i in range(n_languages):
            lang_negative_ctxs = []
            if n_negatives:
                tries = 0
                while len(lang_negative_ctxs) < n_negatives:
                    rand_idx = random.randint(0, len(passage_keys) - 1)
                    rand_key = passage_keys[rand_idx]
                    rand_passage = passages[rand_key]

                    if a.lower() not in rand_passage["text"].lower():
                        lang_negative_ctxs.append(rand_key)

                    tries += 1

                    if tries > 20:
                        break

            negative_ctx_ids.append(lang_negative_ctxs)

        pair["negative_ctx_ids"] = negative_ctx_ids

        filtered_passages[passage_id] = passages[passage_id]
        filtered_pairs.append(pair)

    if save_files:
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "filtered_data")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        passages_path = os.path.join(output_dir, "passages.json")
        with open(passages_path, "w") as fp:
            json.dump(filtered_passages, fp)
        print("Wrote passages to file:", passages_path)

        tpassages_path = os.path.join(output_dir, "trimmed_passages.json")
        with open(tpassages_path, "w") as fp:
            json.dump(trimmed_passages, fp)
        print("Wrote passages to file:", tpassages_path)

        pairs_path = os.path.join(output_dir, "qa_pairs.jsonl")
        with jsonlines.open(pairs_path, "w") as writer:
            writer.write_all(filtered_pairs)
        print("Wrote QA pairs to file:", pairs_path)

    return filtered_pairs, filtered_passages


def main(args):
    assert os.path.exists(args.input_dir)
    dirs = sorted([os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)])

    print("Aggregating all pairs and passages...")
    all_pairs, passages, trimmed_passages = aggregate_all(dirs, args.trim_len)
    print("Done.")
    print(
        "Length of all pairs:",
        len(all_pairs),
        "\nLength of all passages:",
        len(passages),
    )

    print("Computing valid pairs...")
    valid_idxs = compute_valid_pairs(all_pairs)
    print("Done.")

    filtered_pairs, filtered_passages = get_valid_data(
        valid_idxs,
        all_pairs,
        passages,
        trimmed_passages,
        n_negatives=args.n_negatives,
        n_languages=args.n_languages,
        save_files=args.save_files,
        output_dir=args.output_dir,
    )
    print(
        "Length of valid pairs:",
        len(filtered_pairs),
        "\nLength of passages of valid pairs:",
        len(filtered_passages),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", help="Input directory containing subdirectories."
    )
    parser.add_argument("-o", "--output-dir", help="Output directory.")
    parser.add_argument(
        "-tl",
        "--trim-len",
        default=100,
        type=int,
        help="Checks whether answer is contained in the length of tokens specified (from start or end).",
    )
    parser.add_argument(
        "-nn", "--n-negatives", default=5, type=int, help="Number of negative contexts."
    )
    parser.add_argument(
        "-nl",
        "--n-languages",
        default=16,
        type=int,
        help="Number of languages to create negative contexts for.",
    )
    parser.add_argument(
        "-s",
        "--save-files",
        help="Save files to output directory.",
        action="store_true",
    )
    parser.add_argument("-c", "--clean", help="Clean text.", action="store_true")
    # parser.add_argument('-fs', '--filters', default=["is_wiki", "is_number", "has_number", "is_who_q", "is_how_many_q", "has_date"], nargs="+",help="Filters to apply.")
    args = parser.parse_args()

    main(args)
