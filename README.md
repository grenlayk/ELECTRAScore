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
electra_score_extended_chatgpt         |  0.420827   |   0.344957
electra_score_extended                |   0.414904    |  0.342361
electra_score_extended_chatgpt_mean   |   0.400424   |   0.330969
electra_score                         |   0.399629   |   0.329088
electra_score_mean                    |   0.38889    |   0.318438
electra_score_extended_mean           |   0.388041  |    0.320885
electra_score_extended_chatgpt_min    |   0.385991  |    0.321542
electra_score_extended_min             |  0.381002  |    0.318244
bart_score_cnn_para (best) | 0.378 |
electra_score_min            |            0.374256 |     0.312055
bart_score_cnn_src_hypo       |           0.356303  |    0.291795
electra_score_median           |          0.348027   |   0.2881
electra_score_extended_median   |         0.338259    |  0.280573
electra_score_extended_chatgpt_median |   0.314408     | 0.262911
bart_score_src_hypo                    |  0.247883     | 0.202297
bert_score_f                           |  0.192841     | 0.157257
rouge2_f                   | 0.158797    | 0.128203
rouge1_f                   | 0.114974    | 0.0938505
rougel_f                   | 0.104976    | 0.0841752



In order to evaluate metrics for Newsroom and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset Newsroom
```

Results:

metric                  |   spearman  |  kendalltau
----------------------- | ---------- | ------------
bart_score_best_prompt | 0.679 |
bart_score_src_hypo                     | 0.670134    |  0.563899
bart_score_cnn_src_hypo                 | 0.639777    |  0.540041
electra_score_extended_median           | 0.545449    |  0.439608
electra_score_extended_mean             | 0.51121     |  0.406235
electra_score_extended                  | 0.498799    |  0.401015
electra_score_median                    | 0.493861    |  0.395535
electra_score_mean                      | 0.441431    |  0.357829
electra_score_extended_min              | 0.438506    |  0.347084
electra_score                           | 0.424298    |  0.344571
electra_score_extended_chatgpt_median   | 0.409995    |  0.335636
electra_score_extended_chatgpt_mean     | 0.401756    |  0.319755
electra_score_extended_chatgpt          | 0.391315    |  0.321415
electra_score_min                       | 0.339152    |  0.266694
electra_score_extended_chatgpt_min      | 0.321303    |  0.252159
bert_score_f                            | 0.140051    |  0.108461
rouge1_f                                | 0.103553    |  0.081811
rougel_f                                | 0.064634    |  0.055283
rouge2_f                                | 0.047841    |  0.032464


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
