gpu=$1
add1=$2
add2=$3
add3=$4
add4=$5

CUDA_VISIBLE_DEVICES=$gpu python run_t5.py \
    --output_dir="./save/mlm/all-wiki-16" \
    --model_type="mt5" \
    --model_name_or_path="google/mt5-base" \
    --train_file="/work-ceph/wifo3/wiki/all16/concat_train.txt" \
    --validation_file="/work-ceph/wifo3/wiki/all16/concat_test.txt" \
    --cache_dir="/work-ceph/wifo3/transformers" \
    --max_seq_length=1000 \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=6 \
    --gradient_accumulation_steps=4 \
    --do_train \
    --do_eval \
    --line_by_line \
    --adafactor \
    --learning_rate=1e-5 \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="10000" \
    --eval_steps="2500" \
    --evaluation_strategy epoch \
    --save_strategy epoch\
    --load_best_model_at_end \
    --num_train_epochs 20 \
    --save_total_limit 1