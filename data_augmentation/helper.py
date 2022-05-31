from unicodedata import normalize

import pysbd
from nltk import sent_tokenize
from unidecode import unidecode

"""
  This function takes in an input of arbitrary length text and computes the minimal amount of times
  we have to split this text in order to satisfy these two constraints
    1) Text is never split mid-sentence
    2) No individual split is larger than 512 tokens long
"""


def create_splits(text, tokenizer, clean):
    lang = "en"
    if lang:
        seg = pysbd.Segmenter(language=lang, clean=clean)
        sents = seg.segment(text)
    else:
        sents = sent_tokenize(text)
    enc_sents_len = [len(tokenizer.encode(s)[:-1]) for s in sents]
    num_splits = 0
    epsilon = 0.00001

    while True:
        num_splits += 1
        splits = []
        i = 0.0

        # Distribute the sentences as equally as possible among the num_splits
        while i < len(sents) - epsilon:
            splits.append((int(i), int(i + (len(sents) / num_splits) + epsilon)))
            i += len(sents) / num_splits

        # If every split in this set is <512 tokens long, keep it. Otherwise keep looping
        splits_len = [sum(enc_sents_len[s:e]) + 1 for (s, e) in splits]
        if all(s < 512 for s in splits_len):
            break
        elif len(sents) == 1:
            print(
                f"WARNING: sentence too long ({len(enc_sents_len[0])} tokens), skipped"
            )
            return []

        # print("SPLITS:", splits)
        if num_splits > 100:
            print(
                f"WARNING: sentence too long ({len(enc_sents_len[0])} tokens) and too many splits ({num_splits}), skipped"
            )
            raise []

    return [" ".join(sents[s:e]) for (s, e) in splits]


def extract_qa_pairs(
    tokenizer, qa, text, clean=False, first_n=None, save_passage=False
):
    text = unidecode(normalize("NFKC", text))
    qa_pairs = {}

    if first_n:
        splits = create_splits(text, tokenizer, clean)[:first_n]
    else:
        splits = create_splits(text, tokenizer, clean)

    for i, split in enumerate(splits):
        if save_passage:
            qa_pairs[str(i)] = {"passage": split, "qa_pairs": qa(split)}
        else:
            qa_pairs[str(i)] = {"qa_pairs": qa(split)}

    return qa_pairs
