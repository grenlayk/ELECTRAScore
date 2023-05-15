# Notes on thesis

This document is created to track thesis's progress, links, sources and etc.

## BARTScore & ELECTRAScore

### Dataset info

We decided to use datasets from [BARTScore](https://github.com/neulab/BARTScore) folders (`SUM/SummEval/data.pkl`, `SUM/Newsroom/data.pkl`) which were placed in `data` folder (`data/SummEval/data.pkl` and `data/Newsroom/data.pkl`).

For each dataset, BARTScore's authors converted the original dataset into an unified form as shown below (all texts tokenized, normal cased). The unified data form is in each dataset folder, and is named as `data.pkl`. Note that there is another file `final_p.pkl` in each dataset folder, which is their calculated score file.

```json
{
    "doc_id": {
        "src": "This is the source text.",
        "ref_summ": "This is the reference summary",
        "sys_summs": {
            "sys_name1": {
                "sys_summ": "This is the system summary.",
                "scores": {
                    "human_metric1": 0.3,
                    "human_metric2": 0.5
                }
            }
        }
    }
}
```

For SummEval dataset, there are multiple references, we have added a field called `ref_summs` as shown below. They take the average when combining multi-reference results.

```json
"ref_summs": [
    "This is the first reference summary.",
    "This is the second reference summary.",
    "..."
]
```

After calculating scores using automatic metrics, the `scores` field for each document is updated, like the one below.
```json
"scores": {
    "auto_metric1": 0.9,
    "auto_metric2": 0.7,
    "human_metric1": 0.3,
    "human_metric2": 0.5
}
```

### Setup

Their trained BARTScore (on ParaBank2) can be downloaded [here](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing). It should be moved to the `models` folder. 

Alternatively, run this code:

```bash
# running from ELECTRAScore root folder
mkdir models
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m 
mv bart_score.pth models/bart_score.pth # downloaded file should be in models folder
```

### Scoring

```bash
python3 score.py --file data/SummEval/data.pkl --output data/SummEval/scores.pkl --device mps:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_extended --electra_score_extended_chatgpt --multi_ref
python3 score.py --file data/Newsroom/data.pkl --output data/Newsroom/scores.pkl --device mps:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_extended --electra_score_extended_chatgpt
```

This code runs model on SummEval and Newsroom datasets, creating `SummEval/scores.pkl` and `Newsroom/scores.pkl` with  `bert_score`, `bart_score`, `bart_score_cnn`, `electra_score`, `electra_score_extended` and `electra_score_extended_chatgpt` scores in `"scores"` field. For SummEval dataset, please add the `--multi_ref` argument.

In order to evaluate metrics for SummEval and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset SummEval
```

Results:


metric                                 |  spearman  |  kendalltau
------------------------------------- | ----------  |------------
electra_score_extended_chatgpt         |  0.421    |   0.345
electra_score_extended                |   0.415    |   0.342
electra_score_extended_chatgpt_mean   |   0.4004   |   0.331
electra_score                         |   0.3996   |   0.329
electra_score_mean                    |   0.389    |   0.318
electra_score_extended_mean           |   0.388    |   0.321
electra_score_extended_chatgpt_min    |   0.386    |   0.322
electra_score_extended_min            |   0.381    |   0.318
bart_score_cnn_para (best bart_score) |   0.378    |
electra_score_min                     |   0.374    |   0.312
bart_score_cnn_src_hypo               |   0.356    |   0.292
electra_score_median                  |   0.348    |   0.288
electra_score_extended_median         |   0.338    |   0.281
electra_score_extended_chatgpt_median |   0.314    |   0.263
bart_score_src_hypo                   |   0.248    |   0.202
bert_score_f                          |   0.193    |   0.157
rouge2_f                              |   0.159    |   0.128
rouge1_f                              |   0.115    |   0.094
rougel_f                              |   0.105    |   0.084



In order to evaluate metrics for Newsroom and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset Newsroom
```

Results:

metric                  |   spearman  |  kendalltau
----------------------- | ---------- | ------------
bart_score_omega_prompt (best bart_score) | 0.679 |
bart_score_src_hypo                      | 0.670    |  0.564
bart_score_cnn_src_hypo                  | 0.640    |  0.540
electra_score_extended_median            | 0.545    |  0.440
electra_score_extended_mean              | 0.511    |  0.406
electra_score_extended                   | 0.499    |  0.401
electra_score_median                     | 0.494    |  0.396
electra_score_mean                       | 0.441    |  0.358
electra_score_extended_min              | 0.439    |  0.347
electra_score                           | 0.424    |  0.345
electra_score_extended_chatgpt_median   | 0.410    |  0.336
electra_score_extended_chatgpt_mean     | 0.402    |  0.320
electra_score_extended_chatgpt          | 0.391    |  0.321
electra_score_min                       | 0.339    |  0.267
electra_score_extended_chatgpt_min      | 0.321    |  0.252
bert_score_f                            | 0.140    |  0.108
rouge1_f                                | 0.104    |  0.082
rougel_f                                | 0.065    |  0.055
rouge2_f                                | 0.049    |  0.032


## Datasets

### Newsroom dataset 

Human evaluation data can be downloaded from [official GitHub repo](https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv). 

Alternative way is usage of BARTScore's `data.pkl` from [Newsroom folder](https://github.com/neulab/BARTScore/tree/main/SUM/Newsroom).

### SummEval dataset

Human annotations for summaries can be downloaded from [official GitHub repo](https://github.com/Yale-LILY/SummEval#human-annotations) as well.

Alternative link to [HuggingFace page](https://huggingface.co/datasets/mteb/summeval).

Another alternative is usage of BARTScore's `data.pkl` from [SummEval folder](https://github.com/neulab/BARTScore/tree/main/SUM/SummEval).

### CoLA dataset

Data can be downloaded from official [web page](https://nyu-mll.github.io/CoLA/). 

Alternative link to [HuggingFace page](https://huggingface.co/datasets/glue/viewer/cola/train):
