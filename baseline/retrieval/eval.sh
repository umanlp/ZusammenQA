SCRIPT_LOCATION=/path/to/MIA-Shared-Task-2022/baseline/mDPR
OUTPUT_DIR=/path/to/retrieval_results

CTX_FILE=/path/to/mia2022_shared_task_all_langs_w100.tsv
INPUT_DIR=...

N_DOCS=100
MODEL=$1

for QUERY_FILE in xorqa/mia_2022_train_data.pkl xorqa/mia_2022_dev_xorqa.pkl xorqa/mia_2022_test_xorqa_without_answers.pkl xorqa/mia2022_test_surprise_tamil_without_answers.pkl xorqa/mia2022_test_surprise_tagalog_without_answers.pkl mkqa_dev/mkqa-ar.pkl mkqa_dev/mkqa-en.pkl mkqa_dev/mkqa-es.pkl mkqa_dev/mkqa-fi.pkl mkqa_dev/mkqa-ja.pkl mkqa_dev/mkqa-km.pkl mkqa_dev/mkqa-ko.pkl mkqa_dev/mkqa-ms.pkl mkqa_dev/mkqa-ru.pkl mkqa_dev/mkqa-sv.pkl mkqa_dev/mkqa-tr.pkl mkqa_dev/mkqa-zh_cn.pkl mkqa_test_without_answers/mkqa-ar.pkl mkqa_test_without_answers/mkqa-en.pkl mkqa_test_without_answers/mkqa-es.pkl mkqa_test_without_answers/mkqa-fi.pkl mkqa_test_without_answers/mkqa-ja.pkl mkqa_test_without_answers/mkqa-km.pkl mkqa_test_without_answers/mkqa-ko.pkl mkqa_test_without_answers/mkqa-ms.pkl mkqa_test_without_answers/mkqa-ru.pkl mkqa_test_without_answers/mkqa-sv.pkl mkqa_test_without_answers/mkqa-tr.pkl mkqa_test_without_answers/mkqa-zh_cn.pkl 
do

    QUERY_DATASET="$(echo $QUERY_FILE | cut -f 1 -d '/')"
    TGT_DIR=$OUTPUT_DIR/$MODEL/$QUERY_DATASET
    FILENAME=$(echo $QUERY_FILE | cut -f 2 -d '/' | cut -f 1 -d '.').jsonl
    TGT_FILE=$TGT_DIR/$FILENAME
    INDEX_DIR=/data/wifo3/faiss-index/$MODEL

    echo parent dir: $QUERY_DATASET
    echo filename: $FILENAME
    echo target dir: $TGT_DIR
    echo target file: $TGT_FILE

    mkdir -p $TGT_DIR
    mkdir -p $INDEX_DIR

    PARAMS="--qa_file data/queries/all/$QUERY_DATASET/$FILENAME \
      --encoded_qa_file $INPUT_DIR/model=$MODEL/$QUERY_FILE \
      --encoded_ctx_file $INPUT_DIR/model=$MODEL/corpus.pkl \
      --ctx_file $CTX_FILE \
      --n-docs $N_DOCS \
      --validation_workers 4 \
      --out_file $TGT_FILE \
      --save_or_load_index \
      --index_path $INDEX_DIR"

    printf "\n\n"
    echo $PARAMS
    printf "\n\n"
    python $SCRIPT_LOCATION/dense_retriever.py $PARAMS

   echo Done with $QUERY_FILE
   echo Results saved to $TGT_FILE

done
