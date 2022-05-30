import os
import lzma
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="input cc-100 file path") #'/work-ceph/wifo3/cc-100/km.txt.xz'
    parser.add_argument('--save_file', type=str, help="save lang file path") #'./langcc/cc_km_500K.txt'
    parser.add_argument('--num_lines', type=int, default=20000, help="number of extracted lines")#300000
    return parser.parse_args()

def is_match(regex, text):
    pattern = re.compile(regex)
    return pattern.search(text, re.IGNORECASE) is not None

def match_num(regex, text):
    pattern = re.compile(regex, re.IGNORECASE)
    return pattern.findall(text)

def store_line(filename):
    if os.path.exists(filename):
        mode = 'a'
    else:
        mode = 'w'
    return mode

if __name__ == '__main__':
    #python langcc_extract.py --input_file="/work-ceph/wifo3/cc100/ru.txt.xz" --save_file="./cc100_ru.txt"
    #python langcc_extract.py --input_file="/work-ceph/wifo3/cc100/te.txt.xz" --save_file="./cc100_te.txt"
    args = parse_args()
    count = 0
    extract_list = []
    num = 0
    with lzma.open(args.input_file, mode='rt') as file:
        for i, line in enumerate(file):
            if len(line.split())>70: ##en(120), km(17), te(90), ru(120), ar(120), fi(120), sv(120), ms(100), bn(45), tr(120), bn(40), es(100), ko(120), tl(150), ta(70)
            #if len(line)>150: ## cn(150), ja(150)
                extract_list.append(line)
                num+=1
            if len(extract_list)>1000:
                mode = store_line(args.save_file)
                with open(args.save_file, mode) as s:
                    for element in extract_list:
                        s.write(element)
                count +=1000
                extract_list = []
            if i%100000==0 and i!=0:
                print("Load {}".format(i))
            if num>args.num_lines:
                print('Num:{}'.format(num))
                break
    mode = store_line(args.save_file)
    if len(extract_list)!=0:
        with open(args.save_file, mode) as s:
            for element in extract_list:
                s.write(element)
    print(num)