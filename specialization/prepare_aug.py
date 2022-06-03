import random
from argparse import ArgumentParser
from tqdm import tqdm
import json
import jsonlines

def load_json(fpath):
    with open(fpath, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def save_json(fpath, data):
    with open(fpath, 'w') as fout:
        json.dump(data , fout)
    # with open(fpath, "w") as f:
    #     f.write("\n".join([json.dumps(d) for d in data]))

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--aug_path", type=str, required=True, default="/work/wifo3/retrieval/tommaso/MIA-Shared-Task-2022/data/train/mdpr/train_data_05-05-2022-11-00-00_100k.json")
    parser.add_argument("--opath", type=str, required=True, default="")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    
    aug_data = load_json(args.aug_path)[0]

    if args.shuffle:
        ##shuffle
        trial = []
        for i in range(0, 1600000, 100000):
            trial += aug_data[i:i+2000]
        print(len(trial))
        new = []
        for i, aug in enumerate(trial):
            if aug['lang']=='zh-cn':
                aug['lang'] = 'zh_cn'
            new.append({})
            new[i]['q_id']=aug['q_id']
            new[i]['question']=aug['question']
            new[i]['answers']=aug['answers']
            new[i]['lang']=aug['lang']
            new[i]['ctxs']=[{}]*(len(aug['negative_ctxs']))

            for j in range(len(aug['negative_ctxs'])):
                pos_ctx_pos = random.randint(0, 2)
                new[i]['ctxs'][j]=aug['negative_ctxs'][j]
                if new[i]['answers'][0] in new[i]['ctxs'][j]['text']:
                    has_answer = True
                else:
                    has_answer = False
                new[i]['ctxs'][j]['has_answer']=has_answer
            pos_ctx_pos = random.randint(0, 2)
            if new[i]['answers'][0] in aug["positive_ctxs"][0]['text']:
                has_answer = True
            else:
                has_answer = False
            aug["positive_ctxs"][0]['has_answer']=has_answer
            new[i]['ctxs'].insert(pos_ctx_pos, aug["positive_ctxs"][0])
        print(len(new))
        save_json(args.opath, new) #"./aug_data/shuffle.json"
    else:
        trial = []
        for i in range(0, 1600000, 100000):
            trial += aug_data[i:i+2000]
        print(len(trial))
        new = []
        for i, aug in enumerate(trial):
            if aug['lang']=='zh-cn':
                aug['lang'] = 'zh_cn'
            new.append({})
            new[i]['q_id']=aug['q_id']
            new[i]['question']=aug['question']
            new[i]['answers']=aug['answers']
            new[i]['lang']=aug['lang']
            new[i]['ctxs']=[{}]*(len(aug['negative_ctxs'])+1)
            for j in range(len(aug['negative_ctxs'])+1):
                if j==0:
                    new[i]['ctxs'][j]=aug['positive_ctxs'][0]
                    new[i]['ctxs'][j]['has_answer']=True
                else:
                    new[i]['ctxs'][j]=aug['negative_ctxs'][j-1]
                    if new[i]['answers'][0] in new[i]['ctxs'][j]['text']:
                        has_answer = True
                    else:
                        has_answer = False
                    new[i]['ctxs'][j]['has_answer']=has_answer
        save_json(args.opath, new) #"/work-ceph/wifo3/chien/seq2seq/aug_data_no_shuffle/no_shuffle.json",