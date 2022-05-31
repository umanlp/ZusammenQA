import os
import uuid
from argparse import ArgumentParser
from datetime import timedelta
from time import time

import jsonlines
from tqdm import tqdm
from translatepy.translators.deepl import DeeplTranslate
from translatepy.translators.google import GoogleTranslateV2

LANG_MAP = {
    "google": {
        "ar": "ar",  # Arabic
        "bn": "bn",  # Bengali
        "en": "en",  # English
        "fi": "fi",  # Finnish
        "ja": "ja",  # Japanese
        "ko": "ko",  # Korean
        "ru": "ru",  # Russian
        "te": "te",  # Telugu
        "es": "es",  # Spanish
        "km": "km",  # Khmer
        "ms": "ms",  # Malay
        "sv": "sv",  # Swedish
        "ta": "ta",  # Tamil
        "tr": "tr",  # Turkish
        "tl": "tl",  # Tagalog
        "zh-cn": "zh-cn",  # Chinese (simplified)
    }
}


SERVICES = {"google": GoogleTranslateV2, "deepl": DeeplTranslate}


def translate(text, src, trg, service):
    try:
        return service.translate(
            text, source_language=src, destination_language=trg
        ).result
    except:
        return ""


def read_jsonlines(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def main(args):
    assert os.path.exists(args.input_file)

    if args.output_dir:
        output_dir = os.path.join(args.output_dir, f"{args.service}")
    else:
        output_dir = os.path.join(os.path.dirname(args.input_file), f"{args.service}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert args.service in SERVICES
    service = SERVICES[args.service]()

    src_lang = "en"

    langs_with = args.langs_with
    langs_without = args.langs_without
    all_langs = [
        LANG_MAP[args.service][l]
        for l in langs_with + langs_without
        if l in LANG_MAP[args.service]
    ]  # normalize language codes
    print(f"Running translations for languages with train data:", langs_with)
    print(f"Running translations for languages without train data:", langs_without)

    data = read_jsonlines(args.input_file)

    if args.end_idx:
        end_idx = args.end_idx
    else:
        end_idx = len(data)

    data = data[args.start_idx : end_idx]

    lang_ids = sorted(list(LANG_MAP[args.service].keys()))

    total_translations = 0  # count all translations
    start_time = time()

    for target_lang in all_langs:
        print(f"Translating for language: {target_lang.upper()}")

        translations = []

        # for d in tqdm(data):
        for i in tqdm(range(0, len(data), args.step_size)):
            step_data = data[i : i + args.step_size]

            lang_idx = lang_ids.index(target_lang)

            questions = []
            answers = []
            passage_ids = []
            types = []
            sentences = []
            passages = []
            negative_ctx_ids = []

            for d in step_data:
                questions.append(d["question"])
                answers.append(d["answer"])
                passage_ids.append(d["passage_id"])
                types.append(d["type"])
                sentences.append(d["answer_sentence"])
                passages.append(d["passage"])
                negative_ctx_ids.append(d["negative_ctx_ids"][lang_idx])

            questions_str = args.join_str.join(questions)
            answers_str = args.join_str.join(answers)

            if src_lang == target_lang:
                questions_t = questions
                answers_t = answers
            else:
                questions_t = translate(
                    questions_str, src_lang, target_lang, service
                ).split(args.join_str)
                answers_t = translate(
                    answers_str, src_lang, target_lang, service
                ).split(args.join_str)

            assert len(questions) == len(questions_t)
            assert len(answers) == len(answers_t)
            assert len(questions) == len(passage_ids) == len(types)

            total_translations += len(questions_t) * 2

            pairs = []
            for q, a, pi, t, s, p, n in zip(
                questions_t,
                answers_t,
                passage_ids,
                types,
                sentences,
                passages,
                negative_ctx_ids,
            ):
                q_id = str(uuid.uuid4())
                pairs.append(
                    {
                        "question": q,
                        "answer": a,
                        "passage_id": pi,
                        "type": t,
                        "q_id": q_id,
                        "answer_sentence": s,
                        "passage": p,
                        "negative_ctx_ids": n,
                    }
                )

            translations.extend(pairs)

        output_f = os.path.join(output_dir, f"qa_pairs_{target_lang}.jsonl")
        with jsonlines.open(output_f, "w") as writer:
            writer.write_all(translations)
        print(f"Saved {target_lang.upper()} translations to files: {output_f}")

        intermediate_time = time() - start_time
        print(
            f"Total translations: {str(total_translations)}\nTotal time: {str(timedelta(seconds=intermediate_time))}\nSeconds/translation: {str(timedelta(seconds=intermediate_time/total_translations))}\nTranslations/second: {str(total_translations/timedelta(seconds=intermediate_time).seconds)}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="The input directory. Its structure should look like this: /input-dir/[XX]/[zz]",
        required=True,
    )
    parser.add_argument("-s", "--service", default="google", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument(
        "-lw",
        "--langs-with",
        default=["ar", "bn", "en", "fi", "ja", "ko", "ru", "te"],
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-lwo",
        "--langs-without",
        default=["es", "km", "ms", "sv", "ta", "tl", "tr", "zh-cn"],
        nargs="+",
        type=str,
    )
    parser.add_argument("-si", "--start-idx", default=0, type=int)
    parser.add_argument("-ei", "--end-idx", type=int)
    parser.add_argument("-ss", "--step-size", default=100, type=int)
    parser.add_argument("-js", "--join-str", default="\n", type=str)
    args = parser.parse_args()

    main(args)
