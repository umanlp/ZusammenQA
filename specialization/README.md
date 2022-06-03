## mT5 Specialization

### 1. Corpora for mt5 language-, domain-specialization

Two datasets **mt5-wiki14** and **mt5-16** are created for intermediate training purpose, in order to encode knowledge via the domain- and language-specific corpus.
You can simply download the data from [mt5-wiki14](https://drive.google.com/drive/folders/1a8b0oH8z3NunrDGOGYgatoja7Ay1qMqH?usp=sharing) and [mt5-16](https://drive.google.com/drive/folders/1KXzX0UdtMV7gns91tnN_1sUND-xWEGn6?usp=sharing). Or you can modify the scripts under `wiki` and `ccnet` folder for your own usage.

### 2. Train mt5 with MLM objective
You can modify the script `run_mlm_t5.sh` with different hyperparameters and datasets for your own usage.


## Different data augmentation variants
### 1. AUG-QA
Here, we only keep the QA translation pairs. And you can simply run the script `prepare_aug.py`. The data can be downloaded from [aug_data](https://drive.google.com/drive/folders/1ZVkZY1H4UIoQiDxvgG_IGZgl_gD5ba7Z?usp=sharing).

### 2. AUG-QAP
Here, we translate both the QA pairs and the extracted passages, and you can further run the script with `prepare_translation.py`. The data can be downloaded from [aug_data_trans](https://drive.google.com/drive/folders/1CBQlNlO296_jnsvKQIEP16xFZySdAw07?usp=sharing).

We use this [translatepy](https://github.com/Animenosekai/translate) as the Translator, where you can `pip install translatepy==2.3` for your own usage.



