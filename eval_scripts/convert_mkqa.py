import jsonlines
import json
import glob

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def load_json(pred_json_file):
    with open(pred_json_file) as prediction_file:
        predictions = json.load(prediction_file)
    return predictions

def load_txt(pred_txt_file):
    with open(pred_txt_file) as f:
        data = f.read().splitlines()
    return data

langs = ['ar', 'en', 'es', 'fi', 'ja', 'km', 'ko', 'ms', 'ru', 'sv', 'tr', 'zh_cn']

for lang in langs:
    #Baseline prediction files provided by MIA-Shared task committee.
    orig_pred_file = "/work/wifo3/MIA-Shared-Task-2022/data/baselines/baseline2/mkqa_baseline2_pred/mkqa_pred_{}.json".format(lang)
    orig_json_predictions = load_json(orig_pred_file)
    if lang=="zh_cn":
        txt_lang = "zh"
        ## Our path for retrieved results
        txt_file = "/work/wifo3/mGEN/baselines/retrieval_results/s2s_format/mkqa/top_3/{}/mgen_output.txt".format(txt_lang)
        gen_txt_predictions = load_txt(txt_file)
    else:
        txt_lang = lang
        ## Our path for retrieved results
        txt_file = "/work/wifo3/mGEN/baselines/retrieval_results/s2s_format/mkqa/top_3/{}/mgen_output.txt".format(txt_lang)
        gen_txt_predictions = load_txt(txt_file)
    print(lang)
    new_dict = {}
    for i, (k, v) in enumerate(orig_json_predictions.items()):
        new_dict[k]=gen_txt_predictions[i]
    ## Specify where you want to store it
    pred_file = "/work/wifo3/MIA-Shared-Task-2022/data/baselines/mgen_result/mkqa/mkqa_pred_{}.json".format(lang)

    save_json(new_dict, pred_file)