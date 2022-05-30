# ZusammenQA: Data Augmentation with Specialized Models for Cross-lingual Open-retrieval Question Answering System

Authors: Chia-Chien Hung, Tommaso Green, Robert Litschko, Tornike Tsereteli, Sotaro Takeshita, Marco Bombieri, Goran Glavaš, Simone Paolo Ponzetto

NAACL 2022. Workshop: Coming soon

## Introduction
This paper introduces our proposed system for the MIA Shared Task on Cross-lingual Open-retrieval Question Answering (COQA). In this challenging scenario, given an input question the system has to gather evidence documents from a multilingual pool and generate from them an answer in the language of the question. We devised several approaches combining different model variants for three main components: *Data Augmentation*, *Passage Retrieval*, and *Answer Generation*. 
For passage retrieval, we evaluated the monolingual BM25 ranker against the ensemble of *re-rankers based on multilingual pretrained language models* (PLMs) and also variants of the shared task baseline, re-training it from scratch using a recently introduced contrastive loss that maintains a strong gradient signal throughout training by means of mixed negative samples.
For answer generation, we focused on language- and domain-specialization by means of continued language model (LM) pretraining of existing multilingual encoders.
Additionally, for both passage retrieval and answer generation, we augmented the training data provided by the task organizers with automatically generated question-answer pairs created from Wikipedia passages to mitigate the issue of data scarcity, particularly for the low-resource languages for which no training data were provided. Our results show that language- and domain-specialization as well as data augmentation help, especially for low-resource languages.

Overview of our proposed framework:

<img src="/img/overview.png" width="1000"/>

Thanks for the organizers from MIA-Shared-Task, our work is mainly modified from [here](https://github.com/mia-workshop/MIA-Shared-Task-2022).

## Citation
If you use any source codes, or datasets included in this repo in your work, please cite the following paper (will be updated soon):
<pre>

</pre>

## Pretrained Models
The pre-trained models can be easily loaded using huggingface [Transformers](https://github.com/huggingface/transformers) or Adapter-Hub [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) library using the **AutoModel** function. Following pre-trained versions are supported:

* `umanlp/mt5-mlm-wiki14`: mt5 pre-trained using MLM objective with concatenation of 14 languages WIKI corpora 
* `umanlp/mt5-mlm-16`: mt5 pre-trained using MLM objective with concatenation of 14 languages WIKI corpora + 2 languages CCNet corpora

Or you can download it from [umanlp/mt5-mlm-wiki14](https://huggingface.co/umanlp/mt5-mlm-wiki14) and [umanlp/mt5-mlm-16](https://huggingface.co/umanlp/mt5-mlm-16)

## Datasets

### 1. Augmented Data 

(Update from Tornike)

### 2. Corpora for mt5 language-, domain-specialization

Two datasets **mt5-wiki14** and **mt5-16** are created for intermediate training purpose, in order to encode knowledge via the domain- and language-specific corpus.
You can simply download the data from [mt5-wiki14]() and [mt5-16](). Or you can modify the scripts under `specialization` folder for your own usage.

## Structure
This repository is currently under the following structure:
```
.
└── baseline
    └── mDPR
    └── mGEN
    └── wikipedia_preprocess 
└── data
└── eval_scripts
└── sample_predictions
└── specialization
└── img
└── README.md
└── orig-README.md
```
