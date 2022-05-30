import MeCab
from collections import Counter
import string
import re
import argparse
import sys
from pythainlp.tokenize import word_tokenize as th_tokenizer
from khmernltk import word_tokenize as km_tokenizer
import jieba.posseg as pseg

wakati = MeCab.Tagger("-Owakati")

lang_dic = {'telugu': 'te', 'swahili': 'sw', 'thai': 'th', 'finnish': 'fi', 'indonesian': 'id',
            'japanese': 'ja', 'russian': 'ru', 'arabic': 'ar', 'english': 'en', 'bengali': 'bn',
            "korean": "ko", "spanish": "es", "hebrew": "he", "swedish": "sv", "danish": "da", "german": "de",
            "hungarian": "hu", "italian": "it", "khmer": "km", "malay": "ms", "dutch": "nl",
            "norwegian": "no", "portuguese": "pt", "turkish": "tr", "vietnamese": "vi", "french": "fr", "polish": "pl",
            "chinese (simplified)": "zh_cn",  "chinese (hong kong)": 'zh_hk', "chinese (traditional)": "zh_tw"}

langs = ['tr', 'hu', 'zh_hk', 'nl', 'ms', 'zh_cn', 'ja', 'de', 'ru', 'pl', 'fi', 'pt', 'km',
         'it', 'fr', 'he', 'vi', 'zh_tw', 'no', 'da', 'th', 'sv', 'es', 'ar', 'en', 'ko', 'en']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="input file path") #"./train/cc100_en_train.txt"
    parser.add_argument('--lang', type=str, help="which language") #"./train/cc100_en_train.txt"
    parser.add_argument('--save_file_name', type=str, default = "./train/train.txt", help = "file name to save")
    return parser.parse_args()


def tokenize_th_text(text):
    tokens = th_tokenizer(text, engine="newmm")
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)

def tokenize_zh_text(text):
    tokens = pseg.cut(text)
    tokens = [w.word for w in tokens]
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)

def tokenize_km_text(text):
    tokens = km_tokenizer(text)
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)


def normalize_answer(s):
    # TODO: should we keep those counter removal?
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))

def save_file(file_name, corpus_list, lang):
    c = 0
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            c+=1
            if lang=="ja":
                s.write("{}".format(element))
            else:
                s.write("{}\n".format(element))
            if i%10000==0:
                print(i)



if __name__ == '__main__':
    #python wiki_tokenize.py --input_file="./wiki_km.txt" --save_file_name="./tokenize/wiki_km_prep.txt" --lang="km"
    args = parse_args()
    with open(args.input_file, 'r') as f:
        data = f.read().split('\n')
        
    final_gts = []
    if args.lang == "ja":
        for i, gt in enumerate(data):
            gt = wakati.parse(gt)
            final_gts.append(gt)
        #final_pred = wakati.parse(pred.replace("・", " ").replace("、", ","))
    elif args.lang == "zh_cn" or args.lang == "zh_hk" or args.lang == "zh_tw":
        for i, gt in enumerate(data):
            gt = tokenize_zh_text(gt[0:2000])
            if i%1000==0:
                print(i)
            final_gts.append(gt)
        #final_pred = tokenize_zh_text(pred)
    elif args.lang == "th":
        for i, gt in enumerate(data):
            gt = tokenize_th_text(gt)
            final_gts.append(gt)
        #final_pred = tokenize_th_text(pred)
    elif args.lang == "km":
        for i, gt in enumerate(data):
            gt = tokenize_km_text(gt)
            final_gts.append(gt)
        #final_pred = tokenize_km_text(pred)
    else:
        print('Lang: {}'.format(args.lang))
        final_gts = data
        #final_pred = pred
    save_file(args.save_file_name, final_gts, args.lang)