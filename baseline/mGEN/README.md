## mGEN

### Baseline
#### 1. Download mDPR retrieval results

These are currently stored under `/work/wifo3/mGEN/baselines/retrieval_results`

- Training data
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_training_dpr_retrieval_results.json
```

- XOR QA development data

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_xorqa_development_dpr_retrieval_results.json
```

- MKQA development data
The retrieval results for MKQA subsets are available here:

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_non_iterative_baselines_mkqa_dev.zip
unzip mia2022_non_iterative_baselines_mkqa_dev.zip
```

#### 2. Convert mDPR output to mGEN train data format
Check `run_seq2seq.sh`

- For training and XORQA Dev

```
python convert_dpr_retrieval_results_to_seq2seq.py \
     --train_fp /work/wifo3/mGEN/baselines/retrieval_results/mia_shared_training_dpr_retrieval_results.json --dev_fp /work/wifo3/mGEN/baselines/retrieval_results/mia_shared_xorqa_development_dpr_retrieval_results.json  \
     --output_dir /work-ceph/wifo3/chien/seq2seq/top_15 \
     --top_n 15 --add_lang
```

- For MKQA dev

zh_cn, ja, ar, ru, tr, sv, ms, km, es, en, ko, fi 
```
python convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp /work/wifo3/mGEN/baselines/retrieval_results/mia2022_non_iterative_baselines_mkqa_dev/mkqa_dev_{lang}_retrieval_baselines_mia_train_non_itertive.json  \
    --output_dir /work-ceph/wifo3/chien/seq2seq/top_15/mkqa/{lang} \
    --top_n 15 --add_lang
```

#### 3. Fine-tuning

- Train `mt5-base` based model

```run_mgen.sh
python finetune_mgen.py \
    --data_dir /path/to/your/data/dir \
    --output_dir /path/to/output/dir \
    --model_name_or_path /path/to/previous_best_checkpoint \
    --model_type mt5 --gpus 8 \
    --do_train \
    --do_predict \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 1000  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 50 \
    --warmup_steps 500 
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
``` 


#### 4. run evaluation
1. Run DPR
TO evaluate your trained mGEN model, you first need to retrieve passages using mDPR. Please follow the instruction in [mDPR](../mDPR) directory.

2. Convert DPR output
Please concert DPR output file as mentioned above.

3. Run mGEN
Please run the mGEN evaluation by running [`eval_mgen.py`](eval_mgen.py).

```
run_eval.sh
```