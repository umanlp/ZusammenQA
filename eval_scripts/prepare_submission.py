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

def load_jsonlines(fpath):
    with open(fpath, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
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

    #1. mlm_all_baseline_top_15_wiki_new
    #2. mlm_all_mdpr_aug_xcse_top_15_wiki_new -> submission_test_mlm_wiki_ck8_mdpr_aug_xcse.json
    #3. mlm_all_mdpr_aug_top_15_wiki_new -> submission_test_mlm_wiki_ck8_mdpr_aug
    #4. mlm_all_ensemble_rank_top_15_wiki_new -> submission_test_mlm_wiki_ck8_ensemble_rank (done)
    #5. mlm_all_baseline_top_15_wiki_new_b1 -> submission_test_mlm_wiki_ck8_b1 (best)
    #6. mlm_all_baseline_top_15_wiki_new_b2 -> submission_test_mlm_wiki_ck8_b2
    #7. mlm_mdpr_aug_newshuffle_top15_wiki_new_b1 -> submission_test_mlm_mdpr_aug_newshuffle_top15_wiki_new_b1
    #8. mlm_mdpr_aug_no_shuffle_top15_wiki_new -> submission_test_mlm_mdpr_aug_noshuffle_top15_wiki_new_b2
    #9. mlm_mdpr_aug_newshuffle_top15_wiki_new -> submission_test_mlm_mdpr_aug_newshuffle_top15_wiki_new_b2
    #10. mlm_all_baseline_top_15_wiki16_new_b1 -> submission_test_mlm_all_baseline_top_15_wiki16_new_b1
    #11. mlm_mdpr_aug_no_shuffle_top15_wiki16_new_trans_b1 -> submission_test_mlm_mdpr_aug_no_shuffle_top15_wiki16_new_trans_b1
    #12. mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_b1 -> mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_b1
    #13. mlm_all_ensemble_rank_top_15_wiki_new -> submission_test_mlm_all_ensemble_rank_top_15_wiki_new
    #14. bm25_oracle -> submission_test_bm25_oracle
    #15. mlm_mdpr_aug_shuffle_top15_wiki16_new_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki16_new_b1
    #16. mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_sd320_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_sd320_b1
    #17. mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_sd628_5e5_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki16_new_trans_sd628_5e5_b1.json
    #18. mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck4_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck4_b1
    #19. mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck6_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck6_b1
    #20. mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck6_bs6_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck6_bs6_b1
    #21. mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b1 -> submission_test_mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b1
    
    ##xorqa-test
    new_dict = {}
    xorqa_txt_file = "/work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2/test/xorqa/mgen_output.txt"
    gen_xorqa_txt_predictions = load_txt(xorqa_txt_file)
    orig_json_file = "/work/wifo3/MIA-Shared-Task-2022/data/eval/mia_2022_test_xorqa_without_answers.jsonl"
    orig_xorqa_json = load_jsonlines(orig_json_file)
    new_dict['xor']={}
    for i in range(len(orig_xorqa_json)):
        new_dict['xor'][orig_xorqa_json[i]['id']]=gen_xorqa_txt_predictions[i]
    
    ###tamil
    xorqa_txt_file = "/work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2/test/xorqa/mgen_output_tamil.txt"
    gen_xorqa_txt_predictions = load_txt(xorqa_txt_file)
    orig_json_file = "/work/wifo3/MIA-Shared-Task-2022/data/eval/mia2022_test_surprise_tamil_without_answers.jsonl"
    orig_xorqa_json = load_jsonlines(orig_json_file)
    new_dict['sup_ta']={}
    for i in range(len(orig_xorqa_json)):
        new_dict['sup_ta'][orig_xorqa_json[i]['id']]=gen_xorqa_txt_predictions[i]

    ###tagalog
    xorqa_txt_file = "/work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2/test/xorqa/mgen_output_tagalog.txt"
    gen_xorqa_txt_predictions = load_txt(xorqa_txt_file)
    orig_json_file = "/work/wifo3/MIA-Shared-Task-2022/data/eval/mia2022_test_surprise_tagalog_without_answers.jsonl"
    orig_xorqa_json = load_jsonlines(orig_json_file)
    new_dict['sup_tl']={}
    for i in range(len(orig_xorqa_json)):
        new_dict['sup_tl'][orig_xorqa_json[i]['id']]=gen_xorqa_txt_predictions[i]

    ###mkqa
    langs = ['ar', 'es', 'ja', 'km', 'ms', 'ru', 'sv', 'tr', 'zh_cn', 'fi', 'ko', 'en'] #, 
    for lang in langs:
        orig_pred_file = "/work/wifo3/MIA-Shared-Task-2022/data/eval/mkqa_test_without_answers/mkqa-{}.jsonl".format(lang)
        orig_mkqa_json = load_jsonlines(orig_pred_file)
        ## Our path for retrieved results
        txt_file = "/work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2/test/mkqa-add2/mgen_output.mkqa.{}.txt".format(lang)
        gen_txt_predictions = load_txt(txt_file)
        print(lang)
        model = 'mkqa_{}'.format(lang)
        new_dict[model]={}
        for i in range(len(orig_mkqa_json)):
            new_dict[model][orig_mkqa_json[i]['id']]=gen_txt_predictions[i]
    save_json(new_dict, "/work/wifo3/MIA-Shared-Task-2022/data/baselines/mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2/submission_test_mlm_mdpr_aug_shuffle_top15_wiki14_new_trans_ck9_b2.json")
    
if __name__ == "__main__":
    main()
