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
python3 score.py --file data/SummEval/data.pkl --output data/SummEval/scores.pkl --device cuda:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_extended --electra_score_extended_chatgpt --multi_ref
python3 score.py --file data/Newsroom/data.pkl --output data/Newsroom/scores.pkl --device cuda:0 --bert_score --bart_score --bart_score_cnn --electra_score --electra_score_extended --electra_score_extended_chatgpt
```

This code runs model on SummEval and Newsroom datasets, creating `SummEval/scores.pkl` and `Newsroom/scores.pkl` with  `bert_score`, `bart_score`, `bart_score_cnn`, `electra_score`, `electra_score_extended` and `electra_score_extended_chatgpt` scores in `"scores"` field. For SummEval dataset, please add the `--multi_ref` argument.

In order to evaluate metrics for SummEval and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset SummEval
```

Results:

```
Human metric: fluency
metric                      spearman    kendalltau
-----------------------  -----------  ------------
bart_score_cnn_src_hypo   0.356487     0.292136
bart_score_src_hypo       0.24802      0.202521
bert_score_f              0.193229     0.157773
rouge2_f                  0.158797     0.128203
rouge1_f                  0.114974     0.0938505
rougel_f                  0.104976     0.0841752
```

ELECTRAScore
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score                  0.388793      0.31634
electra_score_mean             0.355717      0.290085
electra_score_min              0.350164      0.291133
electra_score_median           0.296147      0.243935

olya's
electra_score           0.401178      0.327255
electra_score_min       0.378091      0.315624
electra_score_mean      0.376952      0.308115
electra_score_median    0.288102      0.240813
```

Extended CoLA
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score           0.413197      0.340404
electra_score_mean      0.367449      0.303449
electra_score_min       0.358558      0.301018
electra_score_median    0.331841      0.278291
```

Extended CoLA chatgpt labels
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score           0.416331      0.339389
electra_score_mean      0.408736      0.336604
electra_score_min       0.405182      0.338984
electra_score_median    0.327238      0.27541
```

In order to evaluate metrics for Newsroom and calculate spearman correlation score we can use this code:

```bash
python3 evaluate_stats.py --dataset Newsroom
```

Results:
```
Human metric: fluency
metric                     spearman    kendalltau
-----------------------  ----------  ------------
bart_score_src_hypo       0.670134     0.563899
bart_score_cnn_src_hypo   0.639777     0.540041
bert_score_f              0.140051     0.108461
rouge1_f                  0.103553     0.081811
rougel_f                  0.064634     0.055283
rouge2_f                  0.047841     0.032464
```

ELECTRAScore
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score_median           0.492731      0.401303
electra_score_percentile_75    0.487944      0.395381
electra_score_mean             0.434852      0.361143
electra_score                  0.430446      0.360707
electra_score_percentile_25    0.422495      0.349833
electra_score_min              0.363339      0.29292

olya's:
electra_score_median    0.418112      0.343079
electra_score_mean      0.39058       0.315147
electra_score           0.345181      0.278768
electra_score_min       0.307626      0.247172
```

Extended Cola
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score_median    0.539686      0.439868
electra_score           0.510749      0.40239
electra_score_mean      0.480942      0.375257
electra_score_min       0.407345      0.322078
```

Extended CoLA chatgpt labels
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score_median    0.425211      0.346667
electra_score_mean      0.379102      0.302947
electra_score           0.362951      0.290434
electra_score_min       0.316666      0.250253
```

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
