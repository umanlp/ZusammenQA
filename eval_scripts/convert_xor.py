import jsonlines
import json
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file",
                        default=None, type=str)
    parser.add_argument("--orig_pred_file",
                        default=None, type=str)
    parser.add_argument("--pred_file",
                        default=None, type=str)

    args = parser.parse_args()

    gen_txt_predictions = load_txt(args.txt_file)
    orig_json_predictions = load_json(args.orig_pred_file)
    new_dict = {}
    for i, (k, v) in enumerate(orig_json_predictions.items()):
        new_dict[k]=gen_txt_predictions[i]

    save_json(new_dict, args.pred_file)
    
if __name__ == "__main__":
    #python convert_xor.py --txt_file="/work/wifo3/mGEN/baselines/preds/mgen_output_dev_xor_top_3.txt" --orig_pred_file="/work/wifo3/MIA-Shared-Task-2022/data/baselines/baseline1/xor_dev_output.json" --pred_file="/work/wifo3/MIA-Shared-Task-2022/data/baselines/mgen_result/mgen_output_dev_xor_top_3.json"
    main()
