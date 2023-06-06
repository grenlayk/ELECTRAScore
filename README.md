# ELECTRAScore

This work proposes a new metric called ELECTRAScore, based on the ELECTRA model fine-tuned on the CoLA dataset. We extended the CoLA dataset by employing augmentations and utilizing ChatGPT as a markup supervisor to enhance the correlation between the metric and human evaluation.

## Setup

```bash
pip install -r requirements.txt
```

BARTScore (trained on ParaBank2) can be downloaded [here](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing). It should be moved to the `models` folder. 

Alternatively, run this code:

```bash
# running from ELECTRAScore root folder
mkdir models
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m 
mv bart_score.pth models/bart_score.pth # downloaded file should be in models folder
```

## Data

We use CoLA dataset for fine-tuning. For evaluation we use Newsroom and SummEval datasets. Detailed information about data format and sources can be found in [`data`](data) folder.

## Models and CoLA extension

In our work we use 3 modifications of ELECTRA model:
1. ELECTRA fine-tuned on CoLA dataset ([HuggingFace link](https://huggingface.co/grenlayk/electra-large-cola))
2. ELECTRA fine-tuned on CoLA-E dataset ([HuggingFace link](https://huggingface.co/grenlayk/electra-large-cola-extended))
3. ELECTRA fine-tuned on CoLA-ECL dataset ([HuggingFace link](https://huggingface.co/grenlayk/electra-large-cola-extended-chatgpt))

Where CoLA-E and CoLA-ECL are extended versions of CoLA dataset. More info about these datasets can be found in [cola_augmentations](cola_augmentations) folder. More info about models fine-tuning is in [electra_tuning](electra_tuning) folder.

## How to run

```bash
# SummEval dataset
python3 score.py --file data/SummEval/data.pkl --output data/SummEval/scores.pkl --device cuda:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_e --electra_score_ecl --multi_ref
# Newsroom dataset
python3 score.py --file data/Newsroom/data.pkl --output data/Newsroom/scores.pkl --device cuda:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_e --electra_score_ecl
```

This code runs model on SummEval and Newsroom datasets, creating `SummEval/scores.pkl` and `Newsroom/scores.pkl` with  `bert_score`, `bart_score`, `bart_score_cnn`, `electra_score` (uses model 1), `electra_score_e` (uses model 2) and `electra_score_ecl` (uses model 3) scores in `"scores"` field. For SummEval dataset, please add the `--multi_ref` argument.

In order to evaluate metrics and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset SummEval
```

For SummEval dataset set `--dataset` argument for `SummEval`, for Newsroom dataset set it for `Newsroom`.


## Results

SummEval results:

```
metric                                |  spearman  |  kendalltau
------------------------------------- | ---------- |------------
electra_score_ecl        |    0.421   |   0.345
electra_score_e                |    0.415   |   0.342
electra_score_ecl_mean   |    0.4004  |   0.331
electra_score                         |    0.3996  |   0.329
electra_score_mean                    |    0.389   |   0.318
electra_score_e_mean           |    0.388   |   0.321
electra_score_ecl_min    |    0.386   |   0.322
electra_score_e_min            |    0.381   |   0.318
bart_score_cnn_para (best bart_score) |    0.378   |
electra_score_min                     |    0.374   |   0.312
bart_score_cnn_src_hypo               |    0.356   |   0.292
electra_score_median                  |    0.348   |   0.288
electra_score_e_median         |    0.338   |   0.281
electra_score_ecl_median |    0.314   |   0.263
bart_score_src_hypo                   |    0.248   |   0.202
bert_score_f                          |    0.193   |   0.157
rouge2_f                              |    0.159   |   0.128
rouge1_f                              |    0.115   |   0.094
rougel_f                              |    0.105   |   0.084
```


Newsroom results:
```
metric                                  |  spearman  |  kendalltau
--------------------------------------- | ---------- | ------------
bart_score_best                         |    0.679   |
bart_score_src_hypo                     |    0.670   |  0.564
bart_score_cnn_src_hypo                 |    0.640   |  0.540
electra_score_e_median           |    0.545   |  0.440
electra_score_e_mean             |    0.511   |  0.406
electra_score_e                  |    0.499   |  0.401
electra_score_median                    |    0.494   |  0.396
electra_score_mean                      |    0.441   |  0.358
electra_score_e_min              |    0.439   |  0.347
electra_score                           |    0.424   |  0.345
electra_score_ecl_median   |    0.410   |  0.336
electra_score_ecl_mean     |    0.402   |  0.320
electra_score_ecl          |    0.391   |  0.321
electra_score_min                       |    0.339   |  0.267
electra_score_ecl_min      |    0.321   |  0.252
bert_score_f                            |    0.140   |  0.108
rouge1_f                                |    0.104   |  0.082
rougel_f                                |    0.065   |  0.055
rouge2_f                                |    0.049   |  0.032
```
