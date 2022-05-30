#not_file = '/work/wifo3/data/wikipedia/en/enwiki-20190201-pages-articles-multistream.xml.bz2'
from unicodedata import normalize
import os
import jsonlines
from tqdm.notebook import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type = str, default = "en", help="which language to work on")
    parser.add_argument('--root_data_dir', type=str, default = "/work/wifo3/data/", help="the root directory of data")
    parser.add_argument('--save_file_name', type=str, default = "", help="the filename to save")
    return parser.parse_args()

def read_file(f):
    data = []
    with jsonlines.open(f) as reader:
        for obj in reader:
            data.append(obj)
    return data


def store_line(filename):
    if os.path.exists(filename):
        mode = 'a'
    else:
        mode = 'w'
    return mode

if __name__ == '__main__':
    #!python wiki_extract.py --save_file_name="./wiki_en_trial.txt" --lang="en"
    # bn -> 340
    # ja, zh -> 600
    # en, ar, es, fi, sv, ru, te, tr, ms, ko -> 400
    # km -> 400
    args = parse_args()
    wiki_data_dir = os.path.join(args.root_data_dir, "wikipedia", args.lang)
    all_texts = []
    count = 0
    for subdir in tqdm(sorted([os.path.join(wiki_data_dir,d) for d in os.listdir(wiki_data_dir)])[:-1]):
        files = sorted([os.path.join(subdir,f) for f in os.listdir(subdir)])
        all_texts = []
        for f in files:
            file_txt = []
            data = read_file(f)
            for i, d in enumerate(data):
                #if len(d['text'].split())>200:
                if len(d['text'])>400: #ja, zh, km
                    file_txt.append(normalize('NFKD', d['text']).replace('\n', " ").replace("\'", "'").strip())
            file_txt = list(set(file_txt))
            #print(len(file_txt))
            all_texts+=file_txt
            #count+=len(all_texts)
            if len(all_texts)>1000:
                all_texts = list(set(all_texts))
                print(len(all_texts))
                count+=len(all_texts)
                mode = store_line(args.save_file_name)
                with open(args.save_file_name, mode) as s:
                    for element in all_texts:
                        s.write(element + "\n")
                all_texts = []
        if count>=10000:
            all_texts = list(set(all_texts))
            print(len(all_texts))
            if len(all_texts)!=0:
                mode = store_line(args.save_file_name)
                with open(args.save_file_name, mode) as s:
                    for element in all_texts:
                        s.write(element + "\n")
            break
    all_texts = list(set(all_texts))
    print(len(all_texts))
    with open(args.save_file_name, "r") as s:
        data = s.read().split('\n')
    print(len(set(data)), len(data))
    print(data[0])