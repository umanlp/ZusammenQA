## Install packages
1. No Conda env
```
pip install -r requirements.txt
```

2. Have Conda env
```
source activate /work/wifo3/envs/mgen_shared
conda deactivate
```
## Evaluate predictions locally
### 1. Conversion

#### XORQA
```
python convert_xor.py --txt_file="/work/wifo3/mGEN/baselines/preds/mgen_output_dev_xor_top_3.txt" --orig_pred_file="/work/wifo3/MIA-Shared-Task-2022/data/baselines/baseline1/xor_dev_output.json" --pred_file="/work/wifo3/MIA-Shared-Task-2022/data/baselines/mgen_result/mgen_output_dev_xor_top_3.json"
```

#### MKQA

This differs from the path that you stored on the server, might need to modify a bit regarding the `file path` in the script, now I set it to the baseline model we rerun from mgen with **Top 3** documents.
```
python convert_mkqa.py
```

### 2. Evaluation

Please run the command below to evaluate your models' performance on MKQA and XOR-TyDi QA.

#### XORQA
```
python eval_xor_full.py --data_file "/work/wifo3/MIA-Shared-Task-2022/data/eval/mia_2022_dev_xorqa.jsonl" --pred_file "/work/wifo3/MIA-Shared-Task-2022/data/baselines/mgen_result/mgen_output_dev_xor_top_3.json"
```

#### MKQA
For MKQA, you can run the command above for each language or you can run the command below that takes directory names of the prediction files and input data files.

```
python eval_mkqa_all.py --data_dir "/work/wifo3/MIA-Shared-Task-2022/data/eval/mkqa_dev" --pred_dir "/work/wifo3/MIA-Shared-Task-2022/data/baselines/mgen_result/mkqa" --target ar en es fi ja km ko ms ru tr zh sv

```
You can limit the target languages by setting the `--target` option. You can add the target languages' codes (e.g., `--target en es sw`)

### 3. Prediction

```
python prepare_submission.py
```

## Baseline
The baseline codes are available at [baseline](baseline). 
Our baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models. We have two versions:

1. **multilingual DPR + multilingual seq2seq (CORA without iterative training)**: We train mDPR and mGEN without iterative training process. We first train mDPR, retrieve top passages using the trained mDPR, and then fine-tuned mGEN after we preprocess and augment NQ data using WikiData as in the original CORA paper. Due to the computational costs, we do not re-train the CORA with iterative training on the new data and languages. 

2. **CORA with iterative training**: We run the publicly available CORA's trained models on our evaluation set. We generate dense embeddings for all of the target languages using their mDPR bi encoders as some of the languages (e.g., Chinese - simplified) are not covered by the CORA's original embeddings. There might be some minor differences in data preprocessing of the original CORA paper and our new data. 

### Prediction results of the baselines
We release the final prediction results as well as the intermediate retrieval results for both train and dev sets. 

#### Final Prediction results
- Baseline 1:[MIA2022_Baseline 1 sample_predictions](https://drive.google.com/drive/folders/14Xv6enk7j4d3QKTNbB5jGjaColNffwW_?usp=sharing). 
- Baseline 2: [MIA2022_Baseline 2 sample_predictions](https://drive.google.com/drive/folders/1ePQjLOWUNiF5mr6leAhw8OG-o1h55i75?usp=sharing). 

#### Intermediate Retrieval Results
See the insturctions at [the Baseline's README](https://github.com/mia-workshop/MIA-Shared-Task-2022/tree/main/baseline#intermediate-results----mdpr-retrieval-results). 

#### Final results F1 | EM |
The final results of Baselines 2 and 3 are shown below. The final macro average scores of those baselines are: 
- Baseline 1 = `(38.9 + 18.1 ) / 2`= **28.5** 
- Baseline 2 = `(39.8 + 17.4) / 2`= **28.6** 

- XOR QA 

| Language | (2) F1 | (2) EM |  (1) F1 | (1) EM |  (Ours TOP3) F1 | (Ours TOP3) EM |
| :-----: | :-------:| :------: |  :-------:| :------: |  :-------:| :------: |
| Arabic (`ar`) | 51.3 |  36.0 | 49.7 |  33.7 | 43.9 |31.4|
| Bengali (`bn`) | 28.7 | 20.2 | 29.2 | 21.2 | 24.0|19.0|
| Finnish (`fi`) | 44.4 | 35.7 | 42.7 | 32.9  | 35.8|26.8|
| Japanese (`ja` )| 43.2 | 32.2 | 41.2 | 29.6 | 35.2|25.4|
| Korean (`ko`) |  29.8 | 23.7 | 30.6 | 24.5 | 26.0|20.9|
| Russian (`ru`) |  40.7 | 31.9 | 40.2 | 31.1 | 35.9|28.4|
| Telugu (`te`) |  40.2 | 32.1 | 38.6 | 30.7 | 41.1|34.4|
| Macro-Average |  39.8 | 30.3 | 38.9| 29.1 | 34.6|26.5|

- MKQA

| Language | (2) F1 | (2) EM | (1) F1 | (1) EM |(Ours TOP3) F1 | (Ours TOP3) EM |
| :-----: | :-------:| :------: | :-------:| :------: |  :-------:| :------: |
| Arabic (`ar`) | 8.8  | 5.7 | 8.9 | 5.1 |6.28|4.21|
| English (`en`) | 27.9 | 24.5 | 33.9 | 24.9 |29.6|20.4|
| Spanish (`es`) | 24.9 | 20.9 | 25.1 | 19.3 |21.8|16.6|
| Finnish (`fi`) | 23.3 | 20.0 | 21.1 | 17.4 |17.4|13.7|
| Japanese (`ja`) | 15.2 | 6.3 | 15.3 | 5.8 |10.9|4.5|
| Khmer (`km`) |  5.7 | 4.9 | 6.0 | 4.7 |4.2|3.1|
| Korean (`ko`) |  8.3 | 6.3 |  6.7 | 4.7 |5.2|3.7|
| Malaysian (`ms`) | 22.6  |19.7 | 24.6 | 19.7 |18.7|14.9|
| Russian (`ru`) | 14.0  | 9.4 |  15.6  | 10.6 |10.2|6.3|
| Swedish (`sv`) |  24.1 | 21.1| 25.5 | 20.6|19.8|15.9|
| Turkish (`tr`) |  20.6 | 16.7 | 20.4 |  16.1 |14.1|10.9|
| Chinese-simplified (`zh_cn`) | 13.1  | 6.1 | 13.7  | 5.7 |9.6|4.3|
| Macro-Average  | 17.4 | 13.5 |  18.1  | 12.9 |14.0|9.9|