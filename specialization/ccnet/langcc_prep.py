import random
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import unicodedata
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type = int, default = 10, help="set random seed")
    parser.add_argument('--train_size', type=int, default = 100000, help="size for training set")
    parser.add_argument('--test_size', type=int, default = 10000, help="size for testing set")
    parser.add_argument('--input_file', type=str, help="input file path") #"./domain/taxi_en.txt"
    parser.add_argument('--save_train_file_name', type=str, default = "./train/train.txt", help = "file name for training data")
    parser.add_argument('--save_test_file_name', type=str, default = "./test/test.txt", help = "file name for testing data")
    return parser.parse_args()

def remove_puncts(text):
    #return re.sub('[\.\s*]+', ".", text)
    return re.sub(r"\.+", ".", text)

def remove_email(text):
    text = re.sub(r"\[…\]", " ", text)
    text = re.sub(r"\S*@\S*\s?", "", text)
    return re.sub(r"\_+", " ", text)

def prep_text(text):
    characters = ["", "{", "}", "", "", "", "", "", "⠀"]
    if not any(j for j in text if unicodedata.category(j).startswith('C')) and not any(j in text for j in characters):
        return True
    else:
        return False

def preprocess(text):
    text = text.replace("^", "")
    text = re.sub('_+', ' ', text)
    text = re.sub('\*+', ' ', text)
    text = re.sub(',+', ' ', text)
    text = re.sub('@', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('-+', ' ', text)
    text = re.sub('<+', ' ', text)
    text = text.replace('. . ', ' ')
    if ">" in text[0:4] and "<link>" not in text[0:7]:
        text = re.sub('>', " ", text[0:4]) + text[4: ]
    text = text.strip()
    return text

def remove_url(text):
    text = re.sub(r'\(?\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*\)?', "[URL]", text)
    text = re.sub(r"\(?http\S+\)?", "[URL]", text)
    text = re.sub(r"\(?\w+\.com\)?", "", text)
    text = re.sub(r"www\d{0,3}[.]", "[URL]", text)
    text = re.sub(r'\[(.*?)\]', r'\1', text)
    text = re.sub(r"w*/*@\w+", "", text)
    text = re.sub(r"\S+\s*bit.ly\S+\s*\S+", "[link]", text)
    text.replace("[URL]", " ").replace("URL", " ").replace('[ url ]', '[URL]')
    text = re.sub(r'[/\\]+', " ", text)
    return re.sub(' +', ' ', text) #de, ar, ru, en, ko
    #return re.sub(' +', '', text) #cn, jp

def save_file(file_name, corpus_list, max_len = 10000):
    c = 0
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            if prep_text(element):
                element = remove_email(remove_puncts(remove_url(preprocess(element))))
                #element = re.sub(' +', '', element) #cn, ja
                element = re.sub(' +', ' ', element) #other languages
                element = element.strip()
                if element!="[URL]" and len(element)>100 and c<max_len: #cn, ja
                #if element!="[URL]" and len(element.split())>100 and c<max_len: #other languages
                    c+=1
                    s.write("{}\n".format(element))
                if i%10000==0:
                    print(i)
                
def save_file_url(file_name, corpus_list, max_len = 10000):
    c = 0
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            element = element.replace('[ url ]', '[URL]')
            element = element.strip()
            s.write("{}\n".format(element))
            if i%10000==0:
                print(i)            

if __name__ == '__main__':
    #!python langcc_prep.py --input_file="./cc100_en.txt" --save_train_file_name="./cc100_en_train.txt" --save_test_file_name="./cc100_en_test.txt"
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    with open(args.input_file, 'r') as f:
        data = f.read().split('\n')
        data = data[0:300000]
    print("Original data size: {}".format(len(data)))
    random.shuffle(data)
    train, test = train_test_split(data, test_size=0.001, random_state=args.random_seed)
    train = train[0:args.train_size+200000]
    test = test[0:args.test_size+30000]
    print("Training data size: {}".format(len(train)))
    print("Testing data size: {}".format(len(test)))

    save_file(args.save_test_file_name, test, max_len=args.test_size)
    save_file(args.save_train_file_name, train, max_len=args.train_size)