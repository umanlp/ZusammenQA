from argparse import ArgumentParser
import json
import jsonlines
from translatepy.translators.google import GoogleTranslateV2
import time
        
def load_json(fpath):
    with open(fpath, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def save_json(fpath, data):
    with open(fpath, 'w') as fout:
        json.dump(data , fout)
    # with open(fpath, "w") as f:
    #     f.write("\n".join([json.dumps(d) for d in data]))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--aug_path", type=str, default="./aug_data/shuffle/shuffle.json")
    parser.add_argument("--opath", type=str, default="./aug_data_trans/shuffle/shuffle_trans.json")
    args = parser.parse_args()
    
    aug_data = load_json(args.aug_path)
    gtranslate_v2 = GoogleTranslateV2(service_url="translate.google.com")
    for i, aug_sample in enumerate(aug_data[0]):
        for a, txt in enumerate(aug_data[0][i]['ctxs']):
            if aug_data[0][i]['lang']=="zh_cn":
                lang = "zh"
            else:
                lang = aug_data[0][i]['lang']
            if aug_data[0][i]['lang'] in ['ru','sv','ta','te','tl','tr','zh_cn', 'ar','bn','es','fi','ja','km','ko','ms']:
                aug_data[0][i]['ctxs'][a]['title'] = gtranslate_v2.translate(aug_data[0][i]['ctxs'][a]['title'], source_language= "en", destination_language= lang).result
                aug_data[0][i]['ctxs'][a]['text'] = gtranslate_v2.translate(aug_data[0][i]['ctxs'][a]['text'], source_language= "en", destination_language= lang).result
        if i%1000==0 and i!=0:
            time.sleep(5)
            gtranslate_v2 = GoogleTranslateV2(service_url="translate.google.com")
            print(i)
        if i%2000==0 and i!=0:
            save_json(args.opath, aug_data)
    save_json(args.opath, aug_data)