import argparse
import os

import datasets
import jsonlines
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline as pipelineHF

from helper import create_splits, extract_qa_pairs
from pipeline import pipeline


def print_qa_pairs(qa_pairs):
    for pair in qa_pairs:
        print(pair)


def read_file(path):
    data = None
    format = None
    if ".jsonl" in path:
        data = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                data.append(obj)
        format = "jsonl"
    elif ".txt" in path:
        with open(path, "r") as f:
            data = f.read()
        format = "str"
    else:
        try:
            data = []
            with jsonlines.open(path, "r") as reader:
                for obj in reader:
                    data.append(obj)
            format = "jsonl"
        except:
            raise Exception(f"Unknown file format: {path}")
    return data, format


def make_dataset(data, tokenizer, clean):
    df = pd.DataFrame(data)
    df["text"] = df["text"].apply(lambda x: create_splits(x, tokenizer, clean))
    dataset = datasets.Dataset.from_pandas(df)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--use_summary",
        help="Include summarization pre-processing",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--use_generator",
        help="Include generation pre-processing",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--fast",
        help="Use the smaller and faster versions of the models",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--indir",
        help="The name of the text file to generate questions from. \
                                            If no file is given questions, are generated on user input",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Clean input text before processing. This only works for the following languages: [Arabic, Chinese, English, Japanese, Russian, Spanish]",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--outdir", help="Save generated files to this directory."
    )
    parser.add_argument(
        "-of",
        "--outfile",
        help="The name of the text file to write the generated questions to.",
        type=str,
    )
    parser.add_argument(
        "-sn",
        "--start-n",
        default=0,
        help="Starting index of Wikipedia directories.",
        type=int,
    )
    parser.add_argument(
        "-en", "--end-n", help="Starting index of Wikipedia directories.", type=int
    )
    parser.add_argument(
        "-fn",
        "--first-n",
        help="Generate QA pairs only for the first n text splits.",
        type=int,
    )
    parser.add_argument(
        "-cpu", "--use-cpu", help="Option to use CPU.", action="store_true"
    )
    parser.add_argument(
        "-sp",
        "--save-passage",
        help="Save passage with generated QA pairs.",
        action="store_true",
    )
    args = parser.parse_args()

    qg_model = (
        "valhalla/t5-small-qa-qg-hl" if args.fast else "valhalla/t5-base-qa-qg-hl"
    )
    sum_model = (
        "sshleifer/distilbart-cnn-6-6" if args.fast else "facebook/bart-large-cnn"
    )

    print("Loading QG Model (This may take a while)...")
    use_cuda = True if not args.use_cpu and torch.cuda.is_available() else False
    qg = pipeline("multitask-qa-qg", model=qg_model, use_cuda=use_cuda)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    if args.use_summary:
        print("Loading Summarization Model (This may also take a while)...")

    # if not args.use_cpu and torch.cuda.is_available():
    if use_cuda:
        summarizer = (
            pipelineHF("summarization", model=sum_model, device=0)
            if args.use_summary
            else None
        )
    else:
        summarizer = (
            pipelineHF("summarization", model=sum_model) if args.use_summary else None
        )

    if args.indir:
        assert os.path.exists(args.indir)
        indirs = sorted([os.path.join(args.indir, d) for d in os.listdir(args.indir)])

        assert args.start_n < len(indirs)

        end_n = len(indirs) if not args.end_n else args.end_n
        for indir in tqdm(indirs[args.start_n : end_n]):
            if os.path.isdir(indir):

                if args.outdir:
                    outdir = args.outdir
                    if indir not in outdir:
                        outdir = os.path.join(outdir, os.path.basename(indir))
                else:
                    if args.use_summary:
                        outdir = indir.replace("wikipedia", "wikipedia_aug_sum")
                    elif args.use_generator:
                        outdir = indir.replace("wikipedia", "wikipedia_aug_gen")
                    else:
                        outdir = indir.replace("wikipedia", "wikipedia_aug")

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                files = sorted([os.path.join(indir, f) for f in os.listdir(indir)])

                for f in tqdm(files):
                    outfile = os.path.join(outdir, "qa_" + os.path.basename(f))

                    if os.path.exists(outfile):
                        print(f"Skipping file: {f} as it already exists: {outfile}")
                        continue

                    data, format = read_file(f)
                    print(f"Processing file: {f} in format: {format}")

                    if format == "jsonl":
                        qa_pairs = []
                        for d in tqdm(data):
                            try:
                                text = d["text"]
                                if text != "" and isinstance(text, str):
                                    d["qa_pairs"] = extract_qa_pairs(
                                        tokenizer,
                                        qg,
                                        text,
                                        clean=args.clean,
                                        first_n=args.first_n,
                                        save_passage=args.save_passage,
                                    )
                                    qa_pairs.append(d)
                                else:
                                    print(f"Document with  id:{d['id']} has no text.")
                            except:
                                print(
                                    f"Failed to generate questions for id:{d['id']} in file: {f}"
                                )
                    else:
                        qa_pairs = extract_qa_pairs(
                            tokenizer,
                            qg,
                            data,
                            clean=args.clean,
                            print_summary=False,
                            first_n=args.first_n,
                            save_passage=args.save_passage,
                        )

                    if outfile:
                        with jsonlines.open(outfile, "w") as writer:
                            writer.write_all(qa_pairs)
                        print(f"Wrote QAs to file: {outfile}")
                    else:
                        print_qa_pairs(qa_pairs)
    else:
        while True:
            input_text = input(">")
            qa_pairs = extract_qa_pairs(tokenizer, qg, input_text, args.clean)
            if args.outfile:
                with jsonlines.open(args.outfile, "a") as writer:
                    writer.write_all(qa_pairs)
            else:
                print_qa_pairs(qa_pairs)
