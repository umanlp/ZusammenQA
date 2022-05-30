gpu=$1
# CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
#     --model_name_or_path /work-ceph/wifo3/chien/output/bm25_oracle/checkpoint6/ \
#     --evaluation_set  /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/tagalog/test.source  \
#     --gold_data_path /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/tagalog/gold_para_qa_data_test.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/bm25_oracle_v2/test/xorqa/mgen_output_tagalog.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1
    
# CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
#     --model_name_or_path /work-ceph/wifo3/chien/output/bm25_oracle/checkpoint6/ \
#     --evaluation_set  /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/tamil/test.source  \
#     --gold_data_path /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/tamil/gold_para_qa_data_test.tsv \
#     --predictions_path /work-ceph/wifo3/chien/output/bm25_oracle_v2/test/xorqa/mgen_output_tamil.txt \
#     --gold_data_mode qa \
#     --model_type mt5 \
#     --max_length 20 \
#     --eval_batch_size 1
    
# ## Need to run all the languages one by one?
for lang in "zh_cn" "ja" "ar" "ru" "tr" "sv" "ms" "km" "es" "en" "fi" "ko"
do
CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
    --model_name_or_path /work-ceph/wifo3/chien/output/bm25_oracle/checkpoint6/ \
    --evaluation_set  /work-ceph/wifo3/chien/seq2seq/bm25_oracle/mkqa-test/${lang}/test.source  \
    --gold_data_path /work-ceph/wifo3/chien/seq2seq/bm25_oracle/mkqa-test/${lang}/gold_para_qa_data_test.tsv \
    --predictions_path /work-ceph/wifo3/chien/output/bm25_oracle/test/mkqa/mgen_output.mkqa.${lang}.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1
done

CUDA_VISIBLE_DEVICES=$gpu python eval_mgen.py \
    --model_name_or_path /work-ceph/wifo3/chien/output/bm25_oracle/checkpoint6/ \
    --evaluation_set  /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/test.source  \
    --gold_data_path /work-ceph/wifo3/chien/seq2seq/bm25_oracle/xorqa-test/gold_para_qa_data_test.tsv \
    --predictions_path /work-ceph/wifo3/chien/output/bm25_oracle/test/xorqa/mgen_output.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1