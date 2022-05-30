# CUDA_VISIBLE_DEVICES=7 python finetune_mgen.py \
#     --data_dir /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/mono_bm25_top_15/ \
#     --output_dir /work-ceph/wifo3/chien/output/mono_bm_25_top_15_all/ \
#     --model_name_or_path /work-ceph/wifo3/sft/save/mlm/all/checkpoint-62490 \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

# CUDA_VISIBLE_DEVICES=7 python finetune_mgen.py \
#     --data_dir /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/ \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \
    
# CUDA_VISIBLE_DEVICES=4 python finetune_mgen.py \
#     --data_dir /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/ \
#     --output_dir /work-ceph/wifo3/chien/output/baseline_top_15/ \
#     --model_name_or_path "google/mt5-base" \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \
# 17-6
# CUDA_VISIBLE_DEVICES=6 python finetune_mgen.py \
#     --data_dir /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/ \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#15-0 ensemble_rank with wiki
# CUDA_VISIBLE_DEVICES=0 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/ensemble_rank \
#     --output_dir /work-ceph/wifo3/chien/output/ensemble_rank_mlm_wiki_top_15/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#17-2 ensemble_rank with mt5
# CUDA_VISIBLE_DEVICES=2 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/ensemble_rank \
#     --output_dir /work-ceph/wifo3/chien/output/ensemble_rank_mt5_top_15/ \
#     --model_name_or_path "google/mt5-base" \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#16-0 ensemble_score with wiki
# CUDA_VISIBLE_DEVICES=0 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/ensemble_score \
#     --output_dir /work-ceph/wifo3/chien/output/ensemble_score_mlm_wiki_top_15/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 50 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#15-5 oracle_bm25 with wiki
# CUDA_VISIBLE_DEVICES=5 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/bm25_oracle \
#     --output_dir /work-ceph/wifo3/chien/output/bm25_oracle/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#16-4 mdpr_aug with wiki
# CUDA_VISIBLE_DEVICES=4 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/mdpr_aug \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_mdpr_aug_top_15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#15-0 mdpr (en)
# CUDA_VISIBLE_DEVICES=0 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/mdpr_en \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_mdpr_en_top_15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#dws17-2 Ensemble Rank + MLM wiki
# CUDA_VISIBLE_DEVICES=2 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/ensemble_rank_new \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_ensemble_rank_top_15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#dws17-6 mDPR+Aug+MixCSE + MLM wiki
# CUDA_VISIBLE_DEVICES=6 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/mdpr_aug_xcse \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_mdpr_aug_xcse_top_15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#15-4: mlm_mdpr_aug_shuffle_top15_wiki_new
# CUDA_VISIBLE_DEVICES=4 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/aug_data \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \
#15-7:  mlm_mdpr_aug_no_shuffle_top15_wiki_new
# CUDA_VISIBLE_DEVICES=7 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/aug_data_no_shuffle \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_mdpr_aug_no_shuffle_top15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#16-1: mlm_mdpr_aug_newshuffle_top15_wiki_new
# CUDA_VISIBLE_DEVICES=1 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/aug_data \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_mdpr_aug_newshuffle_top15_wiki_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-new \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#17-5: mlm_mdpr_aug_newshuffle_top15_wiki_new
# CUDA_VISIBLE_DEVICES=5 python finetune_mgen.py \
#     --data_dir /work-ceph/wifo3/chien/seq2seq/aug_data_no_shuffle_trans \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_mdpr_aug_no_shuffle_top15_wiki16_new_trans/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-16/checkpoint-90000 \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \
#17-4
# CUDA_VISIBLE_DEVICES=4 python finetune_mgen.py \
#     --data_dir /work/wifo3/mGEN/baselines/retrieval_results/s2s_format/baseline_top_15/ \
#     --output_dir /work-ceph/wifo3/chien/output/mlm_all_baseline_top_15_wiki16_new/ \
#     --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-16/checkpoint-90000 \
#     --model_type mt5 --gpus 1 \
#     --do_train \
#     --do_predict \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --max_source_length 1000  \
#     --max_target_length 20 \
#     --val_max_target_length 25 \
#     --test_max_target_length 25 \
#     --label_smoothing 0.1 \
#     --dropout 0.1 \
#     --num_train_epochs 22 \
#     --warmup_steps 500 \
#     --learning_rate 3e-05 \
#     --weight_decay 0.001 \
#     --adam_epsilon 1e-08 \
#     --max_grad_norm 0.1 \

#17-6 /work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki16_new_trans
CUDA_VISIBLE_DEVICES=6 python finetune_mgen.py \
    --data_dir /work-ceph/wifo3/chien/seq2seq/aug_data_trans \
    --output_dir /work-ceph/wifo3/chien/output/mlm_mdpr_aug_shuffle_top15_wiki16_new_trans/ \
    --model_name_or_path /work-ceph/wifo3/run_mlm/save/mlm/all-wiki-16/checkpoint-90000 \
    --model_type mt5 --gpus 1 \
    --do_train \
    --do_predict \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --max_source_length 1000  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 22 \
    --warmup_steps 500 \
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \