### For training & XORQA dev
python convert_dpr_retrieval_results_to_seq2seq.py \
    --train_fp /work/wifo3/mGEN/baselines/retrieval_results/mia_shared_training_dpr_retrieval_results.json \
    --aug_fp /work-ceph/wifo3/chien/seq2seq/aug_data_trans/concat_trans.json \
    --output_dir /work-ceph/wifo3/chien/seq2seq/aug_data_trans \
    --top_n 15 --add_lang

### For MKQA dev
# #zh_cn, ja, ar, ru, tr, sv, ms, km, es, en, ko, fi 
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --dev_fp /work/wifo3/mGEN/baselines/retrieval_results/mia2022_non_iterative_baselines_mkqa_dev/mkqa_dev_bn_retrieval_baselines_mia_train_non_itertive.json  \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/top_15/mkqa/bn \
#     --top_n 15 --add_lang
    
## For xorqa
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --train_fp /work-ceph/wifo3/mdpr/retrieval_results/mixcse_adv_aug_unseen_sur/xorqa/mia_2022_train_data_results.json \
#     --dev_fp /work-ceph/wifo3/mdpr/retrieval_results/mixcse_adv_aug_unseen_sur/xorqa/mia_2022_dev_xorqa_results.json  \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/mdpr_aug_xcse \
#     --top_n 15 --add_lang

## For xorqa-dev
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --dev_fp /work/wifo3/mGEN/baselines/retrieval_results/mia_shared_xorqa_development_dpr_retrieval_results.json \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/aug_data_trans \
#     --top_n 15 --add_lang

## For mkqa-dev
# for lang in "zh_cn" "ja" "ar" "ru" "tr" "sv" "ms" "km" "es" "en" "fi" "ko"
# do
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --dev_fp /work-ceph/wifo3/mdpr/retrieval_results/mixcse_adv_aug_unseen_sur/mkqa_dev/mkqa-${lang}_results.json  \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/mdpr_aug_xcse/mkqa-dev/${lang} \
#     --top_n 15 --add_lang
# done

#For xorqa-test -> not working
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --test_fp /work-ceph/wifo3/mdpr/retrieval_results/baseline2/xorqa/mia2022_test_surprise_tamil_without_answers_results.json \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/baseline2/xorqa-test/tamil \
#     --top_n 15 --add_lang

# For mkqa-test
# for lang in "zh_cn" "ja" "ar" "ru" "tr" "sv" "ms" "km" "es" "en" "fi" "ko"
# do
# python convert_dpr_retrieval_results_to_seq2seq.py \
#     --test_fp /work-ceph/wifo3/mdpr/retrieval_results/baseline2/mkqa_test_without_answers/mkqa-${lang}_results.json  \
#     --output_dir /work-ceph/wifo3/chien/seq2seq/baseline2/mkqa-test/${lang} \
#     --top_n 15 --add_lang
# done