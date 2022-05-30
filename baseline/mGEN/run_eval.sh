gpu=$1
# CUDA_VISIBLE_DEVICES=4 python eval_mgen.py \
#     --model_name_or_path /work/stakeshi/PROJECTS/MIA-Shared-Task-2022/baseline/mGEN/output/MONO_BM25_TOP_15/checkpoint35/ \
#     --evaluation_set  /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/mono_bm25_top_15/val.source  \
#     --gold_data_path /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/mono_bm25_top_15/gold_para_qa_data_dev.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/MONO_BM25_TOP_15/mgen_output.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1

## This is for xorqa dev
# CUDA_VISIBLE_DEVICES=3 python eval_mgen.py \
#     --model_name_or_path /work/stakeshi/PROJECTS/MIA-Shared-Task-2022/baseline/mGEN/output/mlm_all_baseline_top_15/checkpoint11/ \
#     --evaluation_set  /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/val.source  \
#     --gold_data_path /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/gold_para_qa_data_dev.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_ck11/mgen_output.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1

# CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
#     --model_name_or_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new/checkpoint8/ \
#     --evaluation_set  /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/val.source  \
#     --gold_data_path /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/gold_para_qa_data_dev.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new_ck8_temp07_topp095/mgen_output.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1


# ## Need to run all the languages one by one?
# for lang in "zh_cn" "ja" "ar" "ru" "tr" "sv" "ms" "km" "es" "en" "fi" "ko"
# do
# CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
#     --model_name_or_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new/checkpoint8/ \
#     --evaluation_set  /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-dev/${lang}/val.source  \
#     --gold_data_path /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-dev/${lang}/gold_para_qa_data_dev.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new_ck8_temp07_topp095/mgen_output.mkqa.${lang}.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1
# done

# for lang in "es" "en" "fi" "ko"
# do
# CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
#     --model_name_or_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new/checkpoint8/ \
#     --evaluation_set  /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-dev/${lang}/val.source  \
#     --gold_data_path /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-dev/${lang}/gold_para_qa_data_dev.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new_ck8/mgen_output.mkqa.${lang}.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1
# done

for lang in "zh_cn"
do
CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
    --model_name_or_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_new/checkpoint8/ \
    --evaluation_set  /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-test/${lang}/test.source  \
    --gold_data_path /work-ceph/wifo3/chien/seq2seq/top_15/mkqa-test/${lang}/gold_para_qa_data_test.tsv \
    --predictions_path /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new_ck8/test/mkqa/mgen_output.mkqa.${lang}.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1
done